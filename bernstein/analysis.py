"""
Содержит функции, которые помогают:
- сравнивать дробь с исходной функцией (строить графики),
- рассчитывать ошибки (через utils.metrics),
- выводить MSE, MAE и т. д.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import compute_basic_metrics

def compare_fraction_vs_function(
    fraction,
    func,
    interval=(0,1),
    num_points=200,
    plot=True
):
    """
    Строит график сравнения (при plot=True),
    выводит метрики: MSE, MAE, MAX_ERR
    """
    x = np.linspace(interval[0], interval[1], num_points)
    y_true = np.array([func(xi) for xi in x])
    y_approx = fraction.evaluate(x)

    metrics = compute_basic_metrics(y_true, y_approx)

    if plot:
        plt.figure()
        plt.plot(x, y_true, label='Исходная функция')
        plt.plot(x, y_approx, label='Бернштейн дробь')
        plt.title(f"MSE={metrics['MSE']:.2e}, MAE={metrics['MAE']:.2e}")
        plt.legend()
        plt.show()

    return metrics

def error_analysis(
    fraction,
    func,
    interval=(0,1),
    num_points=200
):
    """
    Возвращает словарь метрик (MSE, MAE, MAX_ERR).
    """
    x = np.linspace(interval[0], interval[1], num_points)
    y_true = np.array([func(xi) for xi in x])
    y_approx = fraction.evaluate(x)
    return compute_basic_metrics(y_true, y_approx)
