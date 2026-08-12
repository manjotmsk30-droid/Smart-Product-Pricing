"""
Blender: Late-fusion ensemble combiner.
Trained ONLY on OOF predictions to avoid leakage.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import ElasticNet
import lightgbm as lgb


class LateFusionBlender:
    """
    Late-fusion blender that combines predictions from multiple models.
    CRITICAL: Only train on out-of-fold predictions to avoid leakage.
    """
    
    def __init__(self, 
                 model_type='elasticnet',
                 alpha=1e-3,
                 l1_ratio=0.1,
                 random_state=2025):
        """
        Args:
            model_type: 'elasticnet' or 'lgbm'
            alpha: ElasticNet regularization strength
            l1_ratio: ElasticNet L1 ratio
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == 'elasticnet':
            self.model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=random_state,
                max_iter=10000
            )
        elif model_type == 'lgbm':
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 16,
                'learning_rate': 0.05,
                'feature_fraction': 0.7,
                'verbose': -1,
                'random_state': random_state,
            }
            self.model = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit blender on stacked predictions + strong features.
        
        Args:
            X: Stacked predictions from base models + strong raw features
               Expected columns: ['pred_lgbm', 'pred_text', 'pred_image', ...]
            y: Target values (log1p space)
        """
        # CHANGE: Capture ordered feature list at fit time
        # WHY: Enforce exact column order/shape later.
        self.feature_names = X.columns.tolist()

        # CHANGE: Basic input hygiene
        # WHY: Prevent silent NaNs and shape bugs from propagating into the model.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Blender.fit: X must be a pandas DataFrame.")
        if np.isnan(X.values).any():
            raise ValueError("Blender.fit: X contains NaNs. Clean or impute before fitting.")
        y = np.asarray(y, dtype=float).reshape(-1)
        if np.isnan(y).any():
            raise ValueError("Blender.fit: y contains NaNs. Clean your targets.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Blender.fit: X rows ({X.shape[0]}) != y length ({y.shape[0]}).")

        if self.model_type == 'elasticnet':
            self.model.fit(X, y)
            
        elif self.model_type == 'lgbm':
            train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=300,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=100)]
            )
        
        print(f"✓ Fitted {self.model_type} blender with {len(self.feature_names)} features")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using blender.
        
        Args:
            X: Stacked predictions + features (same columns as fit)
            
        Returns:
            Blended predictions in log1p space
        """
        if self.model is None:
            # CHANGE: Guard against unfitted model
            # WHY: Clear error if predict is called before fit/load.
            raise RuntimeError("Blender.predict: model is not fitted. Call fit() or load().")

        if not isinstance(X, pd.DataFrame):
            # CHANGE: Enforce DataFrame
            # WHY: We rely on column names; Series/ndarray is unsafe here.
            raise TypeError("Blender.predict: X must be a pandas DataFrame.")

        # CHANGE: Strict but safe column alignment
        # WHY: Drop unseen columns; add missing with 0.0; reorder to training layout.
        if self.feature_names:
            missing = [c for c in self.feature_names if c not in X.columns]
            extra = [c for c in X.columns if c not in self.feature_names]
            if extra:
                X = X.drop(columns=extra)
            if missing:
                for c in missing:
                    X[c] = 0.0
            X = X[self.feature_names]

        # CHANGE: Guard against NaNs at inference
        # WHY: Defensive fill to avoid downstream model issues.
        if np.isnan(X.values).any():
            X = X.fillna(0.0)
        
        if self.model_type == 'elasticnet':
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def get_feature_weights(self) -> pd.DataFrame:
        """Get feature importance/weights."""
        if self.feature_names is None:
            # CHANGE: Safety check
            # WHY: Provide clearer error if called before fit/load.
            raise RuntimeError("Blender.get_feature_weights: model not fitted/loaded.")

        if self.model_type == 'elasticnet':
            weights = pd.DataFrame({
                'feature': self.feature_names,
                'weight': self.model.coef_
            }).sort_values('weight', key=lambda s: s.abs(), ascending=False)  # CHANGE: robust abs key
            return weights
        else:
            importance = self.model.feature_importance(importance_type='gain')
            weights = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return weights
    
    def save(self, path: str):
        """Save blender."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load blender."""
        data = joblib.load(path)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        return instance


def create_blender_features(pred_lgbm: np.ndarray,
                            pred_text: np.ndarray,
                            pred_image: np.ndarray,
                            strong_features: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create feature DataFrame for blender.
    
    Args:
        pred_lgbm: Predictions from tabular GBDT
        pred_text: Predictions from text head
        pred_image: Predictions from image head
        strong_features: Additional strong raw features (optional)
        
    Returns:
        DataFrame with blender features
    """
    # CHANGE: Ensure 1D arrays for stable DataFrame construction
    # WHY: Avoid accidental (n,1) shapes from some predictors.
    pred_lgbm = np.asarray(pred_lgbm).reshape(-1)
    pred_text = np.asarray(pred_text).reshape(-1)
    pred_image = np.asarray(pred_image).reshape(-1)

    n = len(pred_lgbm)
    if len(pred_text) != n or len(pred_image) != n:
        # CHANGE: Consistency check
        # WHY: Align lengths before assembling features.
        raise ValueError("create_blender_features: prediction arrays must have the same length.")

    blender_df = pd.DataFrame({
        'pred_lgbm': pred_lgbm,
        'pred_text': pred_text,
        'pred_image': pred_image,
    })
    
    # Add interaction features
    blender_df['pred_mean'] = (pred_lgbm + pred_text + pred_image) / 3.0
    # CHANGE: Use np.stack to compute per-row std robustly
    # WHY: Avoids axis confusion with a list of arrays.
    blender_df['pred_std'] = np.std(np.stack([pred_lgbm, pred_text, pred_image], axis=1), axis=1)
    
    # Add strong raw features if provided
    if strong_features is not None:
        if not isinstance(strong_features, pd.DataFrame):
            raise TypeError("create_blender_features: strong_features must be a pandas DataFrame.")
        if len(strong_features) != n:
            raise ValueError("create_blender_features: strong_features length must match predictions.")
        for col in strong_features.columns:
            blender_df[f'raw_{col}'] = strong_features[col].values
    
    return blender_df


if __name__ == "__main__":
    # Test blender
    print("Testing LateFusionBlender...")
    
    np.random.seed(2025)
    n_samples = 200
    
    # Simulate OOF predictions from 3 models
    pred_lgbm = np.random.randn(n_samples) * 0.3 + 5.0
    pred_text = np.random.randn(n_samples) * 0.2 + 5.0
    pred_image = np.random.randn(n_samples) * 0.25 + 5.0
    
    # True target (still in log space for this toy test)
    y_true = 0.4 * pred_lgbm + 0.35 * pred_text + 0.25 * pred_image + np.random.randn(n_samples) * 0.1
    
    # Create strong features
    strong_features = pd.DataFrame({
        'qty_total': np.random.uniform(100, 1000, n_samples),
        'brand_freq': np.random.uniform(0, 100, n_samples),
    })
    
    # Create blender features
    X_blend = create_blender_features(
        pred_lgbm, pred_text, pred_image, strong_features
    )
    
    # Train blender
    blender = LateFusionBlender(model_type='elasticnet')
    blender.fit(X_blend, y_true)
    
    # Predict
    preds = blender.predict(X_blend)
    mse = np.mean((preds - y_true) ** 2)
    print(f"\nBlender MSE: {mse:.4f}")
    
    # Get weights
    weights = blender.get_feature_weights()
    print("\nTop feature weights:")
    print(weights.head())
    
    print("\n✓ Blender test passed")
