"""
Функции для оценки ошибок между y_true и y_pred:
 - compute_basic_metrics => возвращает {'MSE', 'MAE', 'MAX_ERR'}

При необходимости можно расширить RMSE, R^2-score и т.д.
"""

import numpy as np

def compute_basic_metrics(y_true, y_pred):
    """
    Вычисляет несколько метрик ошибки:
     - MSE (mean squared error),
     - MAE (mean absolute error),
     - MAX_ERR (макс. отклонение)

    :param y_true: np.ndarray
    :param y_pred: np.ndarray
    :return: dict
    """
    diff = y_true - y_pred
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))

    return {
        "MSE": mse,
        "MAE": mae,
        "MAX_ERR": max_err
    }
