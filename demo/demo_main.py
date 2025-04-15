"""
demo_main.py

Исправленный демо-скрипт, учитывающий корректный интервал (0..N-1) для 1D- и 2D-сжатия.
Показываем:

1) Аппроксимация sin(x) на [0, pi] (Chebyshev),
2) Аппроксимация cos(x) на [0, pi] (Bernstein),
3) Сжатие 1D гладкого сигнала (sin(2πi/N)+0.3 cos(5πi/N)) c Chebyshev (interval=0..N-1),
4) Сжатие 2D (градиент) c Bernstein (строчное), очень высокая степень (12/12),
5) heavy_func для демонстрации параллели (Bernstein).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Чебышев–Марков
from chebyshev_markov.chebyshev_markov import ChebyshevMarkovFraction
from chebyshev_markov.analysis import compare_fraction_vs_function

# Бернштейн
from bernstein.bernstein import BernsteinFraction
from bernstein.analysis import compare_fraction_vs_function as compare_bernstein

# Интеграция дробей
from core.integration import integrate_fraction

# Сжатие
from utils.compression import (
    compress_signal_1d,
    decompress_signal_1d,
    compress_image_2d,
    decompress_image_2d
)

# Хелперы
from utils.helpers import measure_time
from utils.metrics import compute_basic_metrics
from visualization.plotting import plot_signal_and_approx, plot_2d_data

def heavy_func(x):
    """
    "Тяжёлая" функция (~300k итераций), проверяем параллельное ускорение.
    """
    s = 0.0
    N = 300_000
    for i in range(3):
        s += math.sin(x*(i+1))**2 + math.cos(x/(i+1))**4
    for j in range(N):
        s += math.sqrt( (j+1)*1e-10 + x*x )
    return s

def main():
    # 1) sin(x) c Chebyshev–Марков
    print("=== 1) Аппроксимация sin(x) на [0, pi] (Chebyshev) ===")
    cm_fraction = ChebyshevMarkovFraction.approximate(
        func=math.sin,
        interval=(0, math.pi),
        deg_num=8,
        deg_den=8,
        kind='T',
        poles=None,
        max_iter=15,
        tol=1e-14,
        parallel=True
    )
    metrics_cm = compare_fraction_vs_function(
        cm_fraction, math.sin,
        interval=(0, math.pi),
        num_points=400,
        plot=True
    )
    print("Chebyshev–Markov (sin) metrics:", metrics_cm)
    approx_int = integrate_fraction(cm_fraction, 0, math.pi, n=300)
    print(f"Интеграл sin(x), approx= {approx_int:.6f}, exact=2.000000")

    # 2) cos(x) c Bernstein
    print("\n=== 2) Аппроксимация cos(x) на [0, pi] (Bernstein) ===")
    bern_frac_cos = BernsteinFraction.approximate(
        func=math.cos,
        interval=(0, math.pi),
        deg_num=8,
        deg_den=8,
        poles=None,
        max_iter=15,
        tol=1e-14,
        parallel=False
    )
    metrics_bern_cos = compare_bernstein(
        bern_frac_cos, math.cos,
        interval=(0, math.pi),
        num_points=400,
        plot=True
    )
    print("Bernstein (cos) metrics:", metrics_bern_cos)

    # 3) Сжатие 1D-гладкого сигнала (Chebyshev), interval=(0..N-1)
    print("\n=== 3) Сжатие гладкого 1D (Chebyshev), interval=(0..N-1) ===")
    N = 300
    # Cоздаём x=0..N-1
    x_1d = np.arange(N, dtype=float)

    def smooth_func(i):
        # i: int or float, но округлим внутри compress_signal_1d
        return math.sin(2*math.pi*(i/N)) + 0.3*math.cos(5*math.pi*(i/N))

    # Генерируем сигнал
    signal_1d = np.array([smooth_func(i) for i in x_1d], dtype=float)

    # Сжимаем
    frac_1d, inter_1d = compress_signal_1d(
        signal_1d, x=x_1d,      # x=[0..N-1]
        method='chebyshev',
        deg_num=8,
        deg_den=8
    )
    restored_1d = decompress_signal_1d(frac_1d, inter_1d, N)

    plot_signal_and_approx(
        x_1d, signal_1d, restored_1d,
        title="Сжатие 1D (гладкий), Chebyshev deg=8/8"
    )
    err_1d = compute_basic_metrics(signal_1d, restored_1d)
    print("Ошибки 1D сжатия (smooth):", err_1d)

    # 4) 2D-сжатие (Bernstein), deg=12/12 => почти 1:1
    print("\n=== 4) 2D-сжатие (градиент) построчно (Bernstein, deg=12/12) ===")
    H, W = 80, 60
    image = np.zeros((H,W), dtype=float)
    for i in range(H):
        for j in range(W):
            # Плавный градиент, ~ [0..1.58]
            image[i,j] = i/(H-1) + j/(W-1)

    # degrees=12 => большое, но точнее
    fractions_2d, size_2d = compress_image_2d(
        image, method='bernstein',
        deg_num=12, deg_den=12
    )
    restored_2d = decompress_image_2d(fractions_2d, size_2d)
    plot_2d_data(image, restored_2d, title="2D-sжатие (Bernstein, deg=12/12)")

    diff_2d = image - restored_2d
    mse_2d = np.mean(diff_2d**2)
    print(f"2D MSE= {mse_2d:.8f}")

    # 5) heavy_func
    print("\n=== 5) Аппроксимация heavy_func(0..1) (Bernstein) ===")
    def approximate_heavy(par):
        return BernsteinFraction.approximate(
            func=heavy_func,
            interval=(0,1),
            deg_num=5,
            deg_den=5,
            poles=None,
            max_iter=10,
            tol=1e-8,
            parallel=par
        )

    frac_no_par, t_no_par = measure_time(approximate_heavy, False)
    print(f"heavy_func, parallel=False => time= {t_no_par:.3f} sec")

    frac_par, t_par = measure_time(approximate_heavy, True)
    print(f"heavy_func, parallel=True  => time= {t_par:.3f} sec")

    # Ошибки
    x_test = np.linspace(0,1,50)
    y_true = np.array([heavy_func(xv) for xv in x_test])
    y_np   = frac_no_par.evaluate(x_test)
    y_pp   = frac_par.evaluate(x_test)
    mse_np = np.mean((y_true-y_np)**2)
    mse_pp = np.mean((y_true-y_pp)**2)
    print(f"Bernstein no_parallel MSE= {mse_np:.2e}")
    print(f"Bernstein parallel   MSE= {mse_pp:.2e}")

    print("\n=== Демонстрация завершена ===")

if __name__=="__main__":
    main()
