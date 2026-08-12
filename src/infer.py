"""
Inference pipeline: Generate predictions for test set.
Deterministic, loads all fold models and averages predictions.

Hardening:
- Ensure required quantity keys exist with safe defaults.
- Normalize 'brand' to plain strings (guards against pandas.Categorical).
- Safer blender side-features via reindex + fillna(0).
- Force float64 for prediction buffers and means.
- Cast clip_min to float; guard shapes when stuffing per-fold preds.
- Ensure text_clean is string and non-null.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_seed
from src.preprocess import TextCleaner, QuantityParser, BrandExtractor
from src.fe_text import TextFeatureExtractor, build_text_features
from src.fe_image import ImageFeatureExtractor, build_image_features
from src.models.tabular_gbdt import TabularGBDT, TargetEncoder
from src.models.text_head import TextModelTrainer
from src.models.image_head import ImageModelTrainer
from src.models.fusion_head import LateFusionBlender, create_blender_features


def build_tabular_features(df, qty_parser, brand_extractor, text_col='catalog_content'):
    """Build tabular features (same as training)."""
    features = pd.DataFrame(index=df.index)

    # Parse quantities
    qty_features = df[text_col].apply(qty_parser.parse)
    qty_df = pd.DataFrame(list(qty_features), index=df.index)

    # Ensure required keys exist with safe defaults
    for col, default in (("total_qty_std", 0.0), ("pack_count", 1.0)):
        if col not in qty_df.columns:
            qty_df[col] = default
    qty_df["total_qty_std"] = qty_df["total_qty_std"].fillna(0.0)
    qty_df["pack_count"] = qty_df["pack_count"].fillna(1.0)

    features = pd.concat([features, qty_df], axis=1)

    # Extract brands
    features['brand'] = df[text_col].apply(
        lambda x: brand_extractor.extract(x, x)
    )

    # Add is_bulk flag
    if 'total_qty_std' in features.columns:
        features['is_bulk'] = (
            features['total_qty_std'] > features['total_qty_std'].quantile(0.8)
        ).astype(int)

    return features


def predict_fold(fold_id, df_test, config, artifacts_dir):
    """
    Generate predictions using models from a single fold.

    Returns:
        Dictionary with predictions from each model
    """
    print(f"\nProcessing Fold {fold_id}...")

    fold_dir = Path(artifacts_dir) / f"fold_{fold_id}"

    text_col = 'catalog_content'

    # ===== Preprocessing =====
    cleaner = TextCleaner()
    df_test['text_clean'] = df_test[text_col].apply(cleaner.clean)
    # Ensure text_clean is plain string and non-null
    df_test['text_clean'] = df_test['text_clean'].astype('string').fillna('')

    # ===== Tabular Features =====
    qty_parser = QuantityParser()
    brand_extractor = BrandExtractor()
    X_test_tab = build_tabular_features(df_test, qty_parser, brand_extractor, text_col)

    # Normalize brand dtype to plain string (guards against pandas.Categorical)
    if 'brand' in X_test_tab.columns:
        X_test_tab['brand'] = X_test_tab['brand'].astype('string').fillna('<UNK>')

    # Load and apply target encoder
    target_encoder = TargetEncoder.load(fold_dir / 'target_encoder.pkl')
    X_test_tab['brand_te'] = target_encoder.transform(X_test_tab['brand'], 'brand')

    # Guard sparsity again prior to model/blender use
    X_test_tab['pack_count'] = X_test_tab['pack_count'].fillna(1.0)
    X_test_tab['total_qty_std'] = X_test_tab['total_qty_std'].fillna(0.0)

    # ===== Text Features =====
    text_extractor = TextFeatureExtractor.load(fold_dir / 'text_extractor.pkl')
    tfidf_test, rule_test, _ = build_text_features(
        df_test['text_clean'], text_extractor, fit=False
    )

    # Combine for GBDT
    X_test_combined = pd.concat([
        X_test_tab.reset_index(drop=True),
        pd.DataFrame(tfidf_test, columns=[f'tfidf_{i}' for i in range(tfidf_test.shape[1])]),
        rule_test.reset_index(drop=True)
    ], axis=1)

    # ===== Model 1: LightGBM =====
    lgbm_model = TabularGBDT.load(fold_dir / 'lgbm_model.pkl')
    pred_lgbm = lgbm_model.predict(X_test_combined)

    # ===== Model 2: Text Head =====
    text_model = TextModelTrainer(
        model_name=config['text_model']['model_name'],
        max_length=config['text_model']['max_length']
    )
    text_model.load(fold_dir / 'text_model.pt')
    pred_text = text_model.predict(
        df_test['text_clean'].values,
        batch_size=config['inference']['batch_size']
    )

    # ===== Model 3: Image Head =====
    images_dir = config['paths']['images_dir']
    img_extractor = ImageFeatureExtractor(
        backbone=config['features']['image']['backbone']
    )

    test_embeddings, test_hand = build_image_features(
        df_test['sample_id'], images_dir, img_extractor
    )

    image_model = ImageModelTrainer(
        model_type=config['image_model']['head_type'],
        embedding_dim=img_extractor.embedding_dim
    )
    image_model.load(fold_dir / 'image_model.pkl')
    pred_image = image_model.predict(test_embeddings)

    # ===== Blender =====
    # Safer selection to avoid KeyError and ensure fixed schema
    strong_features = (
        X_test_tab
        .reindex(columns=['total_qty_std', 'pack_count', 'brand_te'])
        .fillna(0)
    )
    X_blend = create_blender_features(
        pred_lgbm, pred_text, pred_image,
        strong_features
    )

    blender = LateFusionBlender.load(fold_dir / 'blender.pkl')
    pred_blend = blender.predict(X_blend)

    return {
        'lgbm': pred_lgbm,
        'text': pred_text,
        'image': pred_image,
        'blend': pred_blend,
    }


def main(config_path: str, test_csv: str, output_csv: str):
    """Main inference pipeline."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['seed'])

    artifacts_dir = Path(config['paths']['artifacts_dir'])

    # Load test data
    print(f"Loading test data from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    print(f"✓ Loaded {len(df_test)} test samples")

    # Validate schema
    required_cols = ['sample_id', 'catalog_content']
    missing = set(required_cols) - set(df_test.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Initialize prediction arrays
    n_folds = config['cv']['n_folds']
    n_samples = len(df_test)

    fold_predictions = {
        'lgbm': np.zeros((n_folds, n_samples), dtype=np.float64),
        'text': np.zeros((n_folds, n_samples), dtype=np.float64),
        'image': np.zeros((n_folds, n_samples), dtype=np.float64),
        'blend': np.zeros((n_folds, n_samples), dtype=np.float64),
    }

    # Get predictions from each fold
    print(f"\nGenerating predictions from {n_folds} folds...")
    for fold_id in range(n_folds):
        fold_preds = predict_fold(fold_id, df_test.copy(), config, artifacts_dir)

        for model_name, preds in fold_preds.items():
            # shape/dtype guard
            arr = np.asarray(preds, dtype=np.float64).reshape(-1)
            fold_predictions[model_name][fold_id, :] = arr

    # ===== Average Predictions Across Folds =====
    print("\nAveraging predictions across folds...")

    # Average in log space
    final_pred_log = np.mean(fold_predictions['blend'].astype(np.float64), axis=0)

    # Convert to price space
    final_pred_price = np.maximum(
        float(config['target']['clip_min']),
        np.expm1(final_pred_log)
    ).astype(np.float64)

    # ===== Create Submission =====
    submission = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': final_pred_price
    })

    # Validation checks
    print("\nValidation checks:")
    print(f"  ✓ Row count: {len(submission)} (expected: {len(df_test)})")
    print(f"  ✓ No NaN values: {not submission['price'].isna().any()}")
    print(f"  ✓ All positive: {(submission['price'] > 0).all()}")
    print(f"  ✓ Price range: [{submission['price'].min():.2f}, {submission['price'].max():.2f}]")

    # Save submission
    submission.to_csv(output_csv, index=False)
    print(f"\n✓ Submission saved to {output_csv}")

    # Show sample predictions
    print("\nSample predictions:")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions on test set')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    parser.add_argument('--test', type=str, default='data/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--output', type=str, default='test_out.csv',
                       help='Path to output CSV')

    args = parser.parse_args()
    main(args.config, args.test, args.output)
