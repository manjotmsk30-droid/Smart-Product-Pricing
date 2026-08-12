"""
Model 1: Tabular GBDT (LightGBM/CatBoost)
Trains on engineered features + TF-IDF SVD components.

CHANGELOG (safe, backward-compatible):
- Added _align_columns(): prevents feature-order drift at predict/inference.
- Hardened unseen-category handling: vectorized mapping; unseen -> -1 (no crashes).
- fit(): works identically with or without external validation.
  * If early_stopping_rounds>0 and no X_val/y_val, it creates a tiny internal holdout to find best_iteration, then REFITS on full data for exactly best_iteration rounds. Deterministic via self.random_state.
  * If no early stopping requested, trains on full data as before.
- TargetEncoder: optional fold_indices for true OOF, leak-safe encodings.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


class TabularGBDT:
    """LightGBM model for tabular features."""
    
    def __init__(self, params: Optional[Dict] = None, random_state: int = 2025):
        """
        Args:
            params: LightGBM parameters
            random_state: Random seed
        """
        self.random_state = random_state
        
        # Default parameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 128,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 20,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,
            'verbose': -1,
            'random_state': random_state,  # LightGBM accepts this alias
            'n_jobs': -1,
        }
        
        if params:
            self.params.update(params)
        
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.categorical_features: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # CHANGE: enforce train-time feature order and shape at fit/predict
    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.feature_names is None:
            return X
        # add missing columns as zeros
        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0
        # drop unexpected columns and reorder
        X = X[self.feature_names]
        return X
    
    def _encode_categoricals(self, X: pd.DataFrame, 
                             cat_columns: list,
                             fit: bool = False) -> pd.DataFrame:
        """
        Label encode categorical columns.
        """
        X = X.copy()
        
        for col in cat_columns:
            if col not in X.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                vals = X[col].astype(str).fillna('missing')
                le.fit(vals)
                self.label_encoders[col] = le
                X[col] = le.transform(vals).astype('int32')
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    col_vals = X[col].astype(str).fillna('missing')
                    # CHANGE: vectorized mapping; unseen -> -1
                    mapping = {cls: i for i, cls in enumerate(le.classes_)}
                    X[col] = col_vals.map(mapping).fillna(-1).astype('int32')
                else:
                    # CHANGE: defensive fallback if encoder missing at predict
                    X[col] = pd.util.hash_pandas_object(
                        X[col].astype(str), index=False
                    ).astype('int64')
        
        return X
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None,
            categorical_features: Optional[list] = None,
            n_estimators: int = 4000,
            early_stopping_rounds: int = 200,
            verbose_eval: int = 100,
            # CHANGE: auto internal holdout when no val is provided
            internal_val_fraction: float = 0.1,
            internal_val_seed: Optional[int] = None) -> Dict:
        """
        Train LightGBM model.

        If X_val/y_val are not provided and early_stopping_rounds>0, an internal
        holdout split is created to determine best_iteration, then the final model
        is refit on the full training data for exactly best_iteration rounds.
        """
        self.feature_names = X_train.columns.tolist()
        self.categorical_features = categorical_features or []

        def _encode_df(df: pd.DataFrame, fit_enc: bool) -> pd.DataFrame:
            enc = self._encode_categoricals(df, self.categorical_features, fit=fit_enc)
            return self._align_columns(enc)

        have_external_val = (X_val is not None) and (y_val is not None)
        want_early_stop = early_stopping_rounds > 0

        if have_external_val:
            # === Regular path with provided validation set ===
            X_train_enc = _encode_df(X_train, fit_enc=True)
            train_data = lgb.Dataset(
                X_train_enc,
                label=y_train,
                feature_name=self.feature_names,
                categorical_feature=[
                    self.feature_names.index(c) for c in self.categorical_features if c in self.feature_names
                ],
            )

            X_val_enc = _encode_df(X_val, fit_enc=False)
            val_data = lgb.Dataset(
                X_val_enc,
                label=y_val,
                reference=train_data,
                feature_name=self.feature_names,
            )

            callbacks = [lgb.log_evaluation(period=verbose_eval)]
            if want_early_stop:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks,
            )

            history = {
                'best_iteration': getattr(self.model, "best_iteration", n_estimators),
                'best_score': getattr(self.model, "best_score", {}),
            }
            return history

        # === No external val provided ===
        if not want_early_stop:
            # Train on full data as before
            X_train_enc = _encode_df(X_train, fit_enc=True)
            train_data = lgb.Dataset(
                X_train_enc,
                label=y_train,
                feature_name=self.feature_names,
                categorical_feature=[
                    self.feature_names.index(c) for c in self.categorical_features if c in self.feature_names
                ],
            )
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=verbose_eval)],
            )
            return {
                'best_iteration': getattr(self.model, "best_iteration", n_estimators),
                'best_score': getattr(self.model, "best_score", {}),
            }

        # CHANGE: early stopping requested but no val set → internal holdout
        from sklearn.model_selection import train_test_split
        rng_seed = self.random_state if internal_val_seed is None else internal_val_seed
        # ensure at least 1 sample in holdout
        ho_size = max(1, int(len(X_train) * internal_val_fraction)) / max(1, len(X_train))
        X_tr, X_ho, y_tr, y_ho = train_test_split(
            X_train, y_train,
            test_size=ho_size,
            random_state=rng_seed,
            shuffle=True,
        )

        # Fit encoders on train split only (no leakage into holdout)
        self.feature_names = X_train.columns.tolist()  # keep original list
        self.label_encoders = {}  # reset to fit on X_tr
        X_tr_enc = _encode_df(X_tr, fit_enc=True)
        train_data = lgb.Dataset(
            X_tr_enc,
            label=y_tr,
            feature_name=self.feature_names,
            categorical_feature=[
                self.feature_names.index(c) for c in self.categorical_features if c in self.feature_names
            ],
        )
        X_ho_enc = _encode_df(X_ho, fit_enc=False)
        val_data = lgb.Dataset(
            X_ho_enc,
            label=y_ho,
            reference=train_data,
            feature_name=self.feature_names,
        )

        callbacks = [
            lgb.log_evaluation(period=verbose_eval),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        ]

        # 1) small model to find best_iteration
        tmp_model = lgb.train(
            self.params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
        )
        best_it = getattr(tmp_model, "best_iteration", n_estimators)

        # 2) REFIT on FULL training data for exactly best_it rounds (no holdout now)
        self.label_encoders = {}
        X_full_enc = _encode_df(X_train, fit_enc=True)
        full_train_data = lgb.Dataset(
            X_full_enc,
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=[
                self.feature_names.index(c) for c in self.categorical_features if c in self.feature_names
            ],
        )
        self.model = lgb.train(
            self.params,
            full_train_data,
            num_boost_round=best_it,
            valid_sets=[full_train_data],
            valid_names=['train'],
            callbacks=[lgb.log_evaluation(period=verbose_eval)],
        )

        history = {
            'best_iteration': best_it,
            'best_score': {'valid': {'rmse': tmp_model.best_score.get('valid', {}).get('rmse', None)}}
        }
        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict on new data.

        Returns predictions in log1p space.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_enc = self._encode_categoricals(X, self.categorical_features, fit=False)
        X_enc = self._align_columns(X_enc)  # CHANGE: enforce same order/shape
        predictions = self.model.predict(
            X_enc, num_iteration=getattr(self.model, "best_iteration", None)
        )
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df
    
    def save(self, path: str):
        """Save model and encoders."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'label_encoders': self.label_encoders,
            'params': self.params,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load saved model."""
        data = joblib.load(path)
        instance = cls(params=data['params'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.categorical_features = data['categorical_features']
        instance.label_encoders = data['label_encoders']
        return instance


class TargetEncoder:
    """
    Fold-safe target encoding for categorical features.
    Prevents leakage by computing statistics only on training folds.
    """
    
    def __init__(self, smoothing: float = 20, min_samples: int = 10):
        """
        Args:
            smoothing: Smoothing parameter (m in formula)
            min_samples: Minimum samples required
        """
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings: Dict[str, Dict[str, float]] = {}
        self.global_mean = 0.0
    
    # CHANGE: optional fold_indices to compute true out-of-fold encodings
    def fit_transform(self, X: pd.Series, y: np.ndarray, 
                      category_col: str,
                      fold_indices: Optional[np.ndarray] = None) -> pd.Series:
        """
        Fit and transform. If fold_indices provided, returns OOF encodings.
        """
        X = X.astype(str)
        self.global_mean = float(np.mean(y))
        
        if fold_indices is None:
            df = pd.DataFrame({'cat': X, 'y': y})
            stats = df.groupby('cat')['y'].agg(['sum', 'count']).reset_index()
            stats['encoded'] = (stats['sum'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            self.encodings[category_col] = dict(zip(stats['cat'], stats['encoded']))
            return X.map(self.encodings[category_col]).fillna(self.global_mean)
        
        # Out-of-fold encoding
        oof_encoded = pd.Series(index=X.index, dtype=float)
        for k in np.unique(fold_indices):
            trn_idx = (fold_indices != k)
            val_idx = (fold_indices == k)
            df_tr = pd.DataFrame({'cat': X[trn_idx], 'y': y[trn_idx]})
            stats = df_tr.groupby('cat')['y'].agg(['sum', 'count']).reset_index()
            stats['encoded'] = (stats['sum'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            mapping = dict(zip(stats['cat'], stats['encoded']))
            oof_encoded.loc[val_idx] = X[val_idx].map(mapping).fillna(self.global_mean)
        
        # Store full-data mapping for test-time transform
        full = pd.DataFrame({'cat': X, 'y': y}).groupby('cat')['y'].agg(['sum', 'count']).reset_index()
        full['encoded'] = (full['sum'] + self.smoothing * self.global_mean) / (full['count'] + self.smoothing)
        self.encodings[category_col] = dict(zip(full['cat'], full['encoded']))
        return oof_encoded
    
    def transform(self, X: pd.Series, category_col: str) -> pd.Series:
        """
        Transform using fitted encodings.
        """
        if category_col not in self.encodings:
            raise ValueError(f"Category {category_col} not fitted")
        return X.astype(str).map(self.encodings[category_col]).fillna(self.global_mean)
    
    def save(self, path: str):
        """Save encodings."""
        joblib.dump({
            'encodings': self.encodings,
            'global_mean': self.global_mean,
            'smoothing': self.smoothing,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load encodings."""
        data = joblib.load(path)
        instance = cls(smoothing=data['smoothing'])
        instance.encodings = data['encodings']
        instance.global_mean = data['global_mean']
        return instance


if __name__ == "__main__":
    # Quick self-test (does not change public API)
    print("Testing TabularGBDT...")
    
    np.random.seed(2025)
    n_samples = 1000
    
    X_train = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
    })
    y_train = np.log1p(100 + 50 * X_train['feature_1'] + 30 * X_train['feature_2'] + 
                       np.random.randn(n_samples) * 10)
    
    X_val = pd.DataFrame({
        'feature_1': np.random.randn(200),
        'feature_2': np.random.randn(200),
        'category_1': np.random.choice(['A', 'B', 'C', 'D'], 200),  # D is unseen
    })
    y_val = np.log1p(100 + 50 * X_val['feature_1'] + 30 * X_val['feature_2'] + 
                     np.random.randn(200) * 10)
    
    # Train with external validation
    model = TabularGBDT()
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        categorical_features=['category_1'],
        n_estimators=200,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    print(f"Best iteration (with val): {history['best_iteration']}")
    
    # Train without external validation (auto internal holdout)
    model2 = TabularGBDT()
    history2 = model2.fit(
        X_train, y_train,
        X_val=None, y_val=None,
        categorical_features=['category_1'],
        n_estimators=200,
        early_stopping_rounds=20,   # triggers internal holdout + refit
        verbose_eval=50
    )
    print(f"Best iteration (no val): {history2['best_iteration']}")
    
    preds = model.predict(X_val)
    print(f"Predictions shape: {preds.shape}")
    
    importance = model.get_feature_importance()
    print("Top 3 features:\n", importance.head(3))
    
    print("✓ TabularGBDT test passed")
