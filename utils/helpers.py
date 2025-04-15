"""
Общие небольшие функции, используемые многими частями библиотеки.
"""

import time
import numpy as np

def generate_points(interval, num=100):
    """
    Генерирует num точек на интервале [xmin, xmax], равномерно.
    """
    xmin, xmax = interval
    return np.linspace(xmin, xmax, num)

def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed
