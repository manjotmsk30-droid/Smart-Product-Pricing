"""
Main training pipeline with 5-fold cross-validation.
Trains all models and blender in a leakage-safe manner.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold  # added KFold
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_seed
from src.metrics import smape, log_smape, evaluate_folds
from src.preprocess import TextCleaner, QuantityParser, BrandExtractor
from src.fe_text import build_text_features, TextFeatureExtractor
from src.fe_image import build_image_features, ImageFeatureExtractor
from src.models.tabular_gbdt import TabularGBDT, TargetEncoder
from src.models.text_head import TextModelTrainer
from src.models.image_head import ImageModelTrainer
from src.models.fusion_head import LateFusionBlender, create_blender_features


def _oof_base_models_for_blender(
    X_tr_combined: pd.DataFrame,
    tr_texts: np.ndarray,
    tr_embeddings: np.ndarray,
    y_tr: np.ndarray,
    config: dict,
    seed: int,
):
    """
    Create TRUE OOF predictions on the train-split only using inner KFold.
    Leakage-safe: brand_te is recomputed INSIDE each inner split.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof_lgbm_tr = np.zeros(len(X_tr_combined), dtype=float)
    oof_text_tr = np.zeros(len(X_tr_combined), dtype=float)
    oof_image_tr = np.zeros(len(X_tr_combined), dtype=float)

    tr_texts = np.asarray(tr_texts)

    for tr_idx, ho_idx in kf.split(X_tr_combined):
        # -----------------------------
        # Recompute brand_te INSIDE the inner split to avoid leakage
        # -----------------------------
        X_tr_fold = X_tr_combined.iloc[tr_idx].copy()
        X_ho_fold = X_tr_combined.iloc[ho_idx].copy()

        # Normalize brand to plain strings (guards against pandas.Categorical)
        if 'brand' in X_tr_fold.columns:
            X_tr_fold['brand'] = X_tr_fold['brand'].astype('string').fillna('<UNK>')
        if 'brand' in X_ho_fold.columns:
            X_ho_fold['brand'] = X_ho_fold['brand'].astype('string').fillna('<UNK>')

        # Drop any precomputed brand_te (computed on full outer-train) to prevent leakage
        for _df in (X_tr_fold, X_ho_fold):
            if 'brand_te' in _df.columns:
                _df.drop(columns=['brand_te'], inplace=True)

        # Fit a fresh TargetEncoder on INNER training portion only, then transform both splits
        te_inner = TargetEncoder(
            smoothing=config['features']['tabular']['target_encoding_smoothing']
        )
        X_tr_fold['brand_te'] = te_inner.fit_transform(
            X_tr_fold['brand'], y_tr[tr_idx], 'brand'
        )
        X_ho_fold['brand_te'] = te_inner.transform(
            X_ho_fold['brand'], 'brand'
        )

        # --- LGBM (trained on leakage-safe inner split) ---
        lgbm_oof = TabularGBDT(params=config['lgbm'])
        lgbm_oof.fit(
            X_tr_fold, y_tr[tr_idx],
            X_ho_fold, y_tr[ho_idx],
            categorical_features=['brand'],
            n_estimators=config['lgbm']['n_estimators'],
            early_stopping_rounds=config['lgbm']['early_stopping_rounds'],
            verbose_eval=config['lgbm']['verbose']
        )
        oof_lgbm_tr[ho_idx] = lgbm_oof.predict(X_ho_fold)

        # --- Text ---
        text_oof = TextModelTrainer(
            model_name=config['text_model']['model_name'],
            max_length=config['text_model']['max_length'],
            dropout=config['text_model']['dropout']
        )
        text_oof.train(
            tr_texts[tr_idx], y_tr[tr_idx],
            tr_texts[ho_idx], y_tr[ho_idx],
            batch_size=config['text_model']['batch_size'],
            epochs_frozen=config['text_model']['epochs_frozen'],
            epochs_unfrozen=config['text_model']['epochs_unfrozen'],
            lr_head=config['text_model']['lr_head'],
            lr_encoder=config['text_model']['lr_encoder'],
            weight_decay=config['text_model']['weight_decay'],
            gradient_clip=config['text_model']['gradient_clip']
        )
        oof_text_tr[ho_idx] = text_oof.predict(
            tr_texts[ho_idx],
            batch_size=config['text_model']['batch_size']
        )

        # --- Image ---
        img_oof = ImageModelTrainer(
            model_type=config['image_model']['head_type'],
            embedding_dim=tr_embeddings.shape[1]
        )
        img_oof.train(
            tr_embeddings[tr_idx], y_tr[tr_idx],
            tr_embeddings[ho_idx], y_tr[ho_idx],
            batch_size=config['image_model']['batch_size'],
            epochs=config['image_model']['epochs'],
            lr=config['image_model']['lr']
        )
        oof_image_tr[ho_idx] = img_oof.predict(tr_embeddings[ho_idx])

    return oof_lgbm_tr, oof_text_tr, oof_image_tr



def create_price_bins(prices, n_bins=20):
    """Create stratification bins from prices."""
    return pd.qcut(prices, q=n_bins, labels=False, duplicates='drop')


def build_tabular_features(df, qty_parser, brand_extractor, text_col='catalog_content'):
    """Build tabular features from dataframe."""
    features = pd.DataFrame(index=df.index)
    
    # Parse quantities
    qty_features = df[text_col].apply(qty_parser.parse)
    qty_df = pd.DataFrame(list(qty_features), index=df.index)
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

# --- sanity guard: all base/blender models output log(price) ---
def _assert_log_space(pred, name):
    m = float(np.nanmean(np.asarray(pred).reshape(-1)))
    # Typical retail log(price) range
    assert 2.0 <= m <= 9.0, f"{name} seems not in log-price space (mean={m:.3f})."

def train_fold(fold_id, train_idx, val_idx, df_train, config, artifacts_dir):
    """
    Train all models for a single fold.
    
    Returns:
        Dictionary with OOF predictions and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_id}")
    print(f"{'='*60}")
    
    fold_dir = Path(artifacts_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    df_tr = df_train.iloc[train_idx].copy()
    df_val = df_train.iloc[val_idx].copy()
    
    y_tr = np.log1p(df_tr['price'].values)
    y_val = np.log1p(df_val['price'].values)
    
    text_col = 'catalog_content'
    
    # ===== Text Preprocessing =====
    print("\n[1/7] Preprocessing text...")
    cleaner = TextCleaner()
    df_tr['text_clean'] = df_tr[text_col].apply(cleaner.clean)
    df_val['text_clean'] = df_val[text_col].apply(cleaner.clean)
    
    # ===== Tabular Features =====
    print("[2/7] Building tabular features...")
    qty_parser = QuantityParser()
    brand_extractor = BrandExtractor()
    
    X_tr_tab = build_tabular_features(df_tr, qty_parser, brand_extractor, text_col)
    X_val_tab = build_tabular_features(df_val, qty_parser, brand_extractor, text_col)

    # normalize brand dtype to plain string; avoids pandas.Categorical edge-cases
    X_tr_tab['brand'] = X_tr_tab['brand'].astype('string').fillna('<UNK>')
    X_val_tab['brand'] = X_val_tab['brand'].astype('string').fillna('<UNK>')

    # ---- Consistent is_bulk threshold computed on TRAIN split ----
# (override any is_bulk that build_tabular_features may have set)
    if 'is_bulk' in X_tr_tab.columns:
        X_tr_tab = X_tr_tab.drop(columns=['is_bulk'])
    if 'is_bulk' in X_val_tab.columns:
        X_val_tab = X_val_tab.drop(columns=['is_bulk'])

# safe defaults in case parser missed values
    X_tr_tab['total_qty_std'] = X_tr_tab['total_qty_std'].fillna(0.0)
    X_val_tab['total_qty_std'] = X_val_tab['total_qty_std'].fillna(0.0)

    bulk_thresh = X_tr_tab['total_qty_std'].quantile(0.8)
    X_tr_tab['is_bulk'] = (X_tr_tab['total_qty_std'] > bulk_thresh).astype(int)
    X_val_tab['is_bulk'] = (X_val_tab['total_qty_std'] > bulk_thresh).astype(int)
# ---- end consistent threshold block ----

    
    # Target encoding for brand
    target_encoder = TargetEncoder(smoothing=config['features']['tabular']['target_encoding_smoothing'])
    X_tr_tab['brand_te'] = target_encoder.fit_transform(X_tr_tab['brand'], y_tr, 'brand')
    X_val_tab['brand_te'] = target_encoder.transform(X_val_tab['brand'], 'brand')
    target_encoder.save(fold_dir / 'target_encoder.pkl')
    
    # ===== Text Features (TF-IDF + SVD) =====
    print("[3/7] Extracting text features...")
    tfidf_tr, rule_tr, text_extractor = build_text_features(
        df_tr['text_clean'], fit=True
    )
    tfidf_val, rule_val, _ = build_text_features(
        df_val['text_clean'], text_extractor, fit=False
    )
    text_extractor.save(fold_dir / 'text_extractor.pkl')
    
    # Combine tabular + text features for GBDT
    X_tr_combined = pd.concat([
        X_tr_tab.reset_index(drop=True),
        pd.DataFrame(tfidf_tr, columns=[f'tfidf_{i}' for i in range(tfidf_tr.shape[1])]),
        rule_tr.reset_index(drop=True)
    ], axis=1)
    
    X_val_combined = pd.concat([
        X_val_tab.reset_index(drop=True),
        pd.DataFrame(tfidf_val, columns=[f'tfidf_{i}' for i in range(tfidf_val.shape[1])]),
        rule_val.reset_index(drop=True)
    ], axis=1)
    
    # ===== Model 1: LightGBM =====
    print("[4/7] Training LightGBM...")
    lgbm_model = TabularGBDT(params=config['lgbm'])
    lgbm_history = lgbm_model.fit(
        X_tr_combined, y_tr,
        X_val_combined, y_val,
        categorical_features=['brand'],
        n_estimators=config['lgbm']['n_estimators'],
        early_stopping_rounds=config['lgbm']['early_stopping_rounds'],
        verbose_eval=config['lgbm']['verbose']
    )
    lgbm_model.save(fold_dir / 'lgbm_model.pkl')
    
    oof_lgbm = lgbm_model.predict(X_val_combined)
    _assert_log_space(oof_lgbm, "LGBM OOF") 
    smape_lgbm = log_smape(y_val, oof_lgbm)
    print(f"  ✓ LightGBM SMAPE: {smape_lgbm:.4f}%")
    
    # ===== Model 2: Text Head =====
    print("[5/7] Training Text Head...")
    text_model = TextModelTrainer(
        model_name=config['text_model']['model_name'],
        max_length=config['text_model']['max_length'],
        dropout=config['text_model']['dropout']
    )
    text_history = text_model.train(
        df_tr['text_clean'].values, y_tr,
        df_val['text_clean'].values, y_val,
        batch_size=config['text_model']['batch_size'],
        epochs_frozen=config['text_model']['epochs_frozen'],
        epochs_unfrozen=config['text_model']['epochs_unfrozen'],
        lr_head=config['text_model']['lr_head'],
        lr_encoder=config['text_model']['lr_encoder'],
        weight_decay=config['text_model']['weight_decay'],
        gradient_clip=config['text_model']['gradient_clip']
    )
    text_model.save(fold_dir / 'text_model.pt')
    
    oof_text = text_model.predict(df_val['text_clean'].values, 
                                   batch_size=config['text_model']['batch_size'])
    _assert_log_space(oof_text, "Text OOF") 
    smape_text = log_smape(y_val, oof_text)
    print(f"  ✓ Text Head SMAPE: {smape_text:.4f}%")
    
    # ===== Model 3: Image Head =====
    print("[6/7] Training Image Head...")
    images_dir = config['paths']['images_dir']
    
    # Extract image features
    img_extractor = ImageFeatureExtractor(
        backbone=config['features']['image']['backbone']
    )
    
    tr_embeddings, tr_hand = build_image_features(
        df_tr['sample_id'], images_dir, img_extractor
    )
    val_embeddings, val_hand = build_image_features(
        df_val['sample_id'], images_dir, img_extractor
    )
    assert tr_embeddings.shape[0] == len(df_tr),  "train image embeddings length mismatch"
    assert val_embeddings.shape[0] == len(df_val), "val image embeddings length mismatch"
    
    # Train image model
    image_model = ImageModelTrainer(
        model_type=config['image_model']['head_type'],
        embedding_dim=img_extractor.embedding_dim
    )
    image_history = image_model.train(
        tr_embeddings, y_tr,
        val_embeddings, y_val,
        batch_size=config['image_model']['batch_size'],
        epochs=config['image_model']['epochs'],
        lr=config['image_model']['lr']
    )
    image_model.save(fold_dir / 'image_model.pkl')
    
    oof_image = image_model.predict(val_embeddings)
    _assert_log_space(oof_image, "Image OOF")
    smape_image = log_smape(y_val, oof_image)
    print(f"  ✓ Image Head SMAPE: {smape_image:.4f}%")
    
    # ===== Blender =====
    print("[7/7] Training Blender...")
    
    # Get strong raw features for blender
    strong_cols = ['total_qty_std', 'pack_count', 'brand_te']
    strong_features_tr = X_tr_tab.reindex(columns=strong_cols).fillna(0)
    strong_features_val = X_val_tab.reindex(columns=strong_cols).fillna(0)

    # TRUE OOF predictions on the train-split using inner KFold (5 splits)
    oof_lgbm_tr, oof_text_tr, oof_image_tr = _oof_base_models_for_blender(
        X_tr_combined=X_tr_combined,
        tr_texts=df_tr['text_clean'].values,
        tr_embeddings=tr_embeddings,
        y_tr=y_tr,
        config=config,
        seed=config['seed'],
    )
    
    # Blender features
    X_blend_tr = create_blender_features(
        oof_lgbm_tr, oof_text_tr, oof_image_tr, strong_features_tr
    )
    X_blend_val = create_blender_features(
        oof_lgbm, oof_text, oof_image, strong_features_val
    )
    
    blender = LateFusionBlender(
        model_type=config['blender']['model_type'],
        alpha=config['blender']['alpha'],
        l1_ratio=config['blender']['l1_ratio']
    )
    blender.fit(X_blend_tr, y_tr)
    blender.save(fold_dir / 'blender.pkl')
    
    oof_blend = blender.predict(X_blend_val)
    _assert_log_space(oof_blend, "Blender OOF")
    smape_blend = log_smape(y_val, oof_blend)
    print(f"  ✓ Blender SMAPE: {smape_blend:.4f}%")
    
    # Return results
    return {
        'oof_predictions': {
            'lgbm': oof_lgbm,
            'text': oof_text,
            'image': oof_image,
            'blend': oof_blend,
        },
        'val_indices': val_idx,
        'metrics': {
            'smape_lgbm': smape_lgbm,
            'smape_text': smape_text,
            'smape_image': smape_image,
            'smape_blend': smape_blend,
        },
        'y_true': y_val,
    }


def main(config_path: str):
    """Main training pipeline."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Create directories
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = Path(config['paths']['runs_dir'])
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df_train = pd.read_csv(config['paths']['train_csv'])
    print(f"✓ Loaded {len(df_train)} training samples")
    
    # Create folds
    print("\nCreating cross-validation folds...")
    price_bins = create_price_bins(df_train['price'], n_bins=config['cv']['stratify_bins'])
    
    skf = StratifiedKFold(
        n_splits=config['cv']['n_folds'],
        shuffle=config['cv']['shuffle'],
        random_state=config['seed']
    )
    
    # Initialize OOF arrays
    n_samples = len(df_train)
    oof_predictions = {
        'lgbm': np.zeros(n_samples),
        'text': np.zeros(n_samples),
        'image': np.zeros(n_samples),
        'blend': np.zeros(n_samples),
    }
    
    fold_metrics = []
    
    # Train each fold
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(df_train, price_bins)):
        fold_results = train_fold(
            fold_id, train_idx, val_idx,
            df_train, config, artifacts_dir
        )
        
        # Store OOF predictions
        for model_name, preds in fold_results['oof_predictions'].items():
            arr = np.asarray(preds, dtype=np.float64).reshape(-1)
            oof_predictions[model_name][val_idx] = arr

        fold_metrics.append(fold_results['metrics'])
    
    # ===== Aggregate Results =====
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}\n")
    
    # Convert to price space and compute final SMAPE
    y_true_price = df_train['price'].values
    
    # use config clip_min and force numeric dtype to avoid object upcast
    target_cfg = config.get('target', {}) or {}
    clip_min = float(target_cfg.get('clip_min', 0.01))

    final_metrics = {}
    for model_name in oof_predictions.keys():
        preds_log = np.asarray(oof_predictions[model_name], dtype=np.float64).reshape(-1)
        oof_price = np.maximum(clip_min, np.expm1(preds_log)).astype(np.float64)
        final_smape = smape(y_true_price, oof_price)
        final_metrics[f'cv_smape_{model_name}'] = final_smape

        
        print(f"{model_name.upper():12s} - CV SMAPE: {final_smape:.4f}%")
    
    # Per-fold breakdown
    print(f"\n{'Model':<12s} {'Mean':<10s} {'Std':<10s} {'Min':<10s} {'Max':<10s}")
    print("-" * 50)
    for model_name in ['lgbm', 'text', 'image', 'blend']:
        scores = [m[f'smape_{model_name}'] for m in fold_metrics]
        print(f"{model_name:<12s} {np.mean(scores):>8.4f}%  {np.std(scores):>8.4f}%  "
              f"{np.min(scores):>8.4f}%  {np.max(scores):>8.4f}%")
    
    # Save results
    results = {
        'config': config,
        'fold_metrics': fold_metrics,
        'final_metrics': final_metrics,
    }
    
    with open(runs_dir / 'cv_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {runs_dir / 'cv_report.json'}")
    print(f"✓ Models saved to {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models with cross-validation')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args.config)
