"""
Evaluation Metrics for Recommender Systems: RMSE and F-Measure.
"""

import numpy as np
from typing import Dict, Tuple
from ..utils.data_loader import RatingDataset


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))


def calc_f_measure(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 4.0) -> float:
    """
    Compute F-Measure (F1 Score) for recommendation utility.
    True relevant: y_true >= threshold
    Predicted relevant: y_pred >= threshold
    """
    actual_positives = (y_true >= threshold)
    pred_positives = (y_pred >= threshold)
    
    tp = np.sum(actual_positives & pred_positives)
    fp = np.sum((~actual_positives) & pred_positives)
    fn = np.sum(actual_positives & (~pred_positives))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return float(2.0 * (precision * recall) / (precision + recall))


def evaluate_model(model, test_dataset: RatingDataset, threshold: float = 4.0) -> Dict[str, float]:
    """
    Evaluate a fitted model on test set ratings.
    """
    preds = model.predict(test_dataset.user_indices, test_dataset.item_indices)
    rmse = calc_rmse(test_dataset.ratings, preds)
    f1 = calc_f_measure(test_dataset.ratings, preds, threshold=threshold)
    return {"rmse": rmse, "f_measure": f1}
