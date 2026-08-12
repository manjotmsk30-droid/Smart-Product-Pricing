"""
SMAPE metric implementation - exact formula from competition.
"""

import numpy as np
from typing import Union


def smape(y_true: Union[np.ndarray, list],
          y_pred: Union[np.ndarray, list],
          eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).

    Formula: mean(|pred - true| / ((|true| + |pred|)/2 + eps)) * 100

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        eps: Small constant to avoid division by zero

    Returns:
        SMAPE score (0-200, lower is better)
    """
    # CHANGED: coerce to contiguous 1-D float arrays to avoid shape/broadcast bugs
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # CHANGED: explicit shape check for clearer errors during CV/infer
    if y_true.shape != y_pred.shape:
        raise ValueError(f"smape: shapes must match, got {y_true.shape} vs {y_pred.shape}")

    numerator = np.abs(y_pred - y_true)
    # Keep competition formula exactly: add eps inside the mean term
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps

    return np.mean(numerator / denominator) * 100.0


def smape_sklearn(y_true, y_pred, eps: float = 1e-8):
    """
    SMAPE wrapper for sklearn compatibility (negative for maximization).
    Used with cross_val_score where higher is better.

    Note: Prefer using sklearn.metrics.make_scorer(smape, greater_is_better=False)
    when passing to GridSearchCV, etc.
    """
    return -smape(y_true, y_pred, eps)


def log_smape(y_true_log: np.ndarray,
              y_pred_log: np.ndarray,
              eps: float = 1e-8) -> float:
    """
    SMAPE on log-transformed predictions.
    Converts back to original scale before computing SMAPE.

    Args:
        y_true_log: log1p(price) ground truth
        y_pred_log: log1p(price) predictions
        eps: Small constant

    Returns:
        SMAPE in original price space
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    # Keep predictions positive to respect pricing domain
    y_pred = np.maximum(0.01, y_pred)

    return smape(y_true, y_pred, eps)


def evaluate_folds(fold_predictions: dict,
                   fold_true_values: dict) -> dict:
    """
    Evaluate SMAPE across multiple folds.

    Args:
        fold_predictions: Dict of {fold_id: predictions}
        fold_true_values: Dict of {fold_id: true_values}

    Returns:
        Dict with per-fold scores and aggregate statistics
    """
    results = {}
    fold_scores = []

    for fold_id in sorted(fold_predictions.keys()):
        y_true = fold_true_values[fold_id]
        y_pred = fold_predictions[fold_id]

        score = smape(y_true, y_pred)
        fold_scores.append(score)
        results[f'fold_{fold_id}'] = score

    results['mean'] = float(np.mean(fold_scores))
    results['std'] = float(np.std(fold_scores))
    results['min'] = float(np.min(fold_scores))
    results['max'] = float(np.max(fold_scores))

    return results


def clip_predictions(predictions: np.ndarray,
                     min_value: float = 0.01) -> np.ndarray:
    """
    Clip predictions to ensure positive prices.

    Args:
        predictions: Raw predictions
        min_value: Minimum allowed price

    Returns:
        Clipped predictions
    """
    return np.maximum(min_value, predictions)


if __name__ == "__main__":
    # Test SMAPE implementation
    y_true = np.array([100, 200, 150, 300])
    y_pred = np.array([120, 180, 150, 280])

    score = smape(y_true, y_pred)
    print(f"SMAPE Test: {score:.2f}%")

    # Expected: roughly 8-9% for these values
    # Manual check: |120-100| / ((100+120)/2) = 20/110 ≈ 18.18%
    #               |180-200| / ((200+180)/2) = 20/190 ≈ 10.53%
    #               |150-150| / ((150+150)/2) = 0
    #               |280-300| / ((300+280)/2) = 20/290 ≈ 6.90%
    # Mean ≈ 8.9%

    assert 8.0 < score < 10.0, f"SMAPE test failed: {score}"
    print("✓ SMAPE implementation verified")
