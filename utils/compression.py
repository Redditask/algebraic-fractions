"""
compression.py

Здесь иллюстрируем "сжатие" 1D и 2D данных (например, сигналов, изображений)
посредством аппроксимации дробями (ChebyshevMarkovFraction или BernsteinFraction).
"""

import numpy as np
from chebyshev_markov.chebyshev_markov import ChebyshevMarkovFraction
from bernstein.bernstein import BernsteinFraction

def compress_signal_1d(signal, x=None, method='chebyshev',
                       deg_num=3, deg_den=3):
    """
    "Сжимает" 1D-сигнал, используя рациональную аппроксимацию заданного типа.
    Возвращает кортеж (fraction, interval), где fraction – объект дроби,
    а interval – (xmin, xmax) для последующего восстановления.

    :param signal: np.ndarray (N, )
        Массив значений сигнала.
    :param x: np.ndarray (N, ) or None
        Координата каждой точки. Если None, берётся [0, 1, 2, ...].
    :param method: str, 'chebyshev' or 'bernstein'
        Тип дроби для аппроксимации.
    :param deg_num: int
        Степень числителя.
    :param deg_den: int
        Степень знаменателя.
    :return: (fraction, interval)
        fraction : ChebyshevMarkovFraction или BernsteinFraction
        interval : (x[0], x[-1])
    """
    N = len(signal)
    if x is None:
        x = np.arange(N)

    interval = (x[0], x[-1])

    # Упрощённый способ интерпретации: округление t -> индекс
    def func(t):
        idx = int(round(t))
        if idx < 0:
            idx = 0
        elif idx >= N:
            idx = N-1
        return float(signal[idx])

    if method == 'chebyshev':
        frac = ChebyshevMarkovFraction.approximate(
            func=func,
            interval=interval,
            deg_num=deg_num,
            deg_den=deg_den,
            kind='T'
        )
    else:
        frac = BernsteinFraction.approximate(
            func=func,
            interval=interval,
            deg_num=deg_num,
            deg_den=deg_den
        )
    return frac, interval

def decompress_signal_1d(fraction, interval, length):
    """
    Восстанавливает 1D-сигнал длины length
    из дроби (fraction) на интервале interval.

    :param fraction: объект дроби (ChebyshevMarkovFraction или BernsteinFraction)
    :param interval: (xmin, xmax)
    :param length: int
    :return: np.ndarray (length, )
    """
    x_new = np.linspace(interval[0], interval[1], length)
    return fraction.evaluate(x_new)

def compress_image_2d(image, method='chebyshev',
                      deg_num=3, deg_den=3):
    """
    "Сжимает" 2D-данные (картинку) построчно, сохраняя
    список дробей для каждой строки.

    :param image: np.ndarray (H, W)
    :param method: str
    :param deg_num: int
    :param deg_den: int
    :return: (fractions, (H, W))
      где fractions = [ (frac_row, interval_row), ... ] по строкам
    """
    H, W = image.shape
    fractions = []
    for h in range(H):
        row = image[h, :]
        frac, inter = compress_signal_1d(
            row, x=np.arange(W),
            method=method,
            deg_num=deg_num,
            deg_den=deg_den
        )
        fractions.append((frac, inter))
    return fractions, (H, W)

def decompress_image_2d(fractions, size):
    """
    Восстанавливает 2D-данные из списка (fraction, interval) для каждой строки.

    :param fractions: list of (fraction, interval)
    :param size: (H, W)
    :return: np.ndarray shape (H, W)
    """
    (H, W) = size
    restored = np.zeros((H, W), dtype=float)
    for h in range(H):
        frac, inter = fractions[h]
        row_approx = decompress_signal_1d(frac, inter, W)
        restored[h, :] = row_approx
    return restored
