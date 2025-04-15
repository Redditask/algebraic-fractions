"""
Содержит базовые численные методы:
 - solve_least_squares: задача МНК
 - numerical_integration, numerical_derivative: простые примеры интегрирования/дифференцирования
"""

import numpy as np

def solve_least_squares(A, b):
    """
    Решает задачу МНК A*x = b, возвращая x, которое минимизирует ||A*x - b||^2.

    :param A: np.ndarray (m, n)
    :param b: np.ndarray (m,)
    :return: np.ndarray (n,) -- решение
    """
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x

def numerical_integration(func, a, b, n=1000):
    """
    Пример простой функции интегрирования (метод трапеций),
    может применяться и для fraction.evaluate, и для любых других.

    :param func: callable, f(x)
    :param a: float, нижняя граница
    :param b: float, верхняя граница
    :param n: int, количество разбиений
    :return: float, оценка интеграла \int_a^b func(x) dx
    """
    x = np.linspace(a, b, n+1)
    y = func(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2.0*np.sum(y[1:-1]) + y[-1])

def numerical_derivative(func, x, h=1e-5):
    """
    Приближённая производная через центральную разность:
      f'(x) ~ (f(x+h) - f(x-h)) / 2h

    :param func: callable
    :param x: float
    :param h: float, шаг
    :return: float
    """
    return (func(x + h) - func(x - h)) / (2.0 * h)
