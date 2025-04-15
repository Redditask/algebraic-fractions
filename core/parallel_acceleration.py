"""
Здесь демонстрируем параллелизацию через Numba (JIT) или joblib.
Можно выносить сюда отдельные вспомогательные функции.

Пример:
 - @njit(parallel=True) для ускорения генерации матриц,
 - joblib.Parallel(...) для распараллеливания большого цикла.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def polynomial_powers_parallel(x_values, degree):
    """
    Генерирует матрицу A (N x (degree+1)),
    где A[i,k] = (x_values[i])^k,
    используя параллельные циклы Numba.

    :param x_values: np.ndarray (N,)
    :param degree: int
    :return: np.ndarray shape (N, degree+1)
    """
    N = x_values.shape[0]
    A = np.zeros((N, degree+1), dtype=np.float64)
    for i in prange(N):  # параллельный цикл
        xi = x_values[i]
        value = 1.0
        A[i, 0] = value
        for k in range(1, degree+1):
            value *= xi
            A[i, k] = value
    return A
