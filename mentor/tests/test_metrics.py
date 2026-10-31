"""
Unit tests for evaluation metrics.
"""

import numpy as np
from src.eval.metrics import calc_rmse, calc_f_measure


def test_rmse_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert calc_rmse(y_true, y_pred) == 0.0
    
    y_pred_offset = y_true + 1.0
    assert np.isclose(calc_rmse(y_true, y_pred_offset), 1.0)


def test_f_measure_computation():
    y_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    y_pred = np.array([4.5, 4.2, 2.0, 1.5, 1.0])
    # Relevant >= 4.0: actual positives indices [0, 1]
    # pred positives indices [0, 1] -> perfect Precision = 1.0, Recall = 1.0 -> F-Measure = 1.0
    assert calc_f_measure(y_true, y_pred, threshold=4.0) == 1.0
