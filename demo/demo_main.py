"""
demo_main.py

Демо-скрипт: аппроксимация, сжатие и параллельное ускорение.

1) Аппроксимация sin(x) и сложных функций на [0, π] (Chebyshev и Bernstein),
2) Сжатие 1D-сигнала (Chebyshev, интервал 0..N-1),
3) Сжатие 2D-изображения с градиентами (Bernstein),
4) Проверка производительности на тяжёлой функции (Bernstein + параллель).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Аппроксимации
from chebyshev_markov.chebyshev_markov import ChebyshevMarkovFraction
from chebyshev_markov.analysis import compare_fraction_vs_function

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

# Хелперы и визуализация
from utils.helpers import measure_time
from utils.metrics import compute_basic_metrics
from visualization.plotting import plot_signal_and_approx, plot_2d_data


def heavy_func(x):
    """
    Тяжёлая функция для теста параллельных вычислений.
    """
    s = 0.0
    N = 300_000
    for i in range(3):
        s += math.sin(x * (i + 1)) ** 2 + math.cos(x / (i + 1)) ** 4
    for j in range(N):
        s += math.sqrt((j + 1) * 1e-10 + x * x)
    return s


def main():
    # === 1. Аппроксимация функций на [0, π] с помощью дробей Чебышева–Маркова ===
    def test_func1(x): return math.sin(x) + 0.5 * math.sin(2 * x) + 0.25 * math.sin(3 * x)

    print("=== 1.1) f₁(x) = sin(x) + 0.5·sin(2x) + 0.25·sin(3x) ===")
    cm1 = ChebyshevMarkovFraction.approximate(
        func=test_func1, interval=(0, math.pi), deg_num=8, deg_den=8,
        kind='T', poles=None, max_iter=15, tol=1e-14, parallel=True
    )
    metrics1 = compare_fraction_vs_function(cm1, test_func1, interval=(0, math.pi), num_points=400)
    print("Метрики f₁:", metrics1)

    approx_I1 = integrate_fraction(cm1, 0, math.pi, n=400)
    print(f"Интеграл f₁, approx = {approx_I1:.6f}, exact = 2.000000\n")

    def test_func2(x):
        gauss = math.exp(-((x - math.pi / 2) ** 2) / (2 * 0.4 ** 2))
        return gauss + 0.3 * math.cos(3 * x)

    print("=== 1.2) f₂(x) = exp(...) + 0.3·cos(3x) ===")
    cm2 = ChebyshevMarkovFraction.approximate(
        func=test_func2, interval=(0, math.pi), deg_num=8, deg_den=8,
        kind='T', poles=None, max_iter=15, tol=1e-14, parallel=True
    )
    metrics2 = compare_fraction_vs_function(cm2, test_func2, interval=(0, math.pi), num_points=400)
    print("Метрики f₂:", metrics2)

    approx_I2 = integrate_fraction(cm2, 0, math.pi, n=400)
    print(f"Интеграл f₂, approx ≈ {approx_I2:.6f}")

    # === 2. Аппроксимации с Bernstein-дробями ===
    def test_func3(x): return math.cos(x) + 0.5 * math.cos(3 * x) + 0.3 * math.sin(5 * x)

    print("\n=== 2.1) f₃(x) = cos(x) + 0.5·cos(3x) + 0.3·sin(5x) ===")
    bern3 = BernsteinFraction.approximate(
        func=test_func3, interval=(0, math.pi), deg_num=10, deg_den=10,
        poles=None, max_iter=20, tol=1e-14, parallel=False
    )
    metrics3 = compare_bernstein(bern3, test_func3, interval=(0, math.pi), num_points=500)
    print("Метрики f₃:", metrics3)

    def test_func4(x): return math.exp(-0.3 * x) * math.sin(4 * x) + 0.2 * math.cos(7 * x)

    print("\n=== 2.2) f₄(x) = exp(-0.3x)·sin(4x) + 0.2·cos(7x) ===")
    bern4 = BernsteinFraction.approximate(
        func=test_func4, interval=(0, math.pi), deg_num=18, deg_den=18,
        poles=None, max_iter=25, tol=1e-14, parallel=False
    )
    metrics4 = compare_bernstein(bern4, test_func4, interval=(0, math.pi), num_points=500)
    print("Метрики f₄:", metrics4)

    # === 3. Сжатие сложного 1D-сигнала (Chebyshev) ===
    print("\n=== 3) Сжатие 1D сигнала ===")
    N = 300
    x_1d = np.arange(N, dtype=float)

    def complex_signal(i):
        t = i / N
        return (
            0.5 * math.sin(2 * math.pi * t)
            + 0.3 * math.cos(5 * math.pi * t)
            + 0.2 * math.sin(8 * math.pi * t) ** 2
            + 0.001 * ((i - N / 2) ** 2) / N
        )

    signal_1d = np.array([complex_signal(i) for i in x_1d])
    frac_1d, inter_1d = compress_signal_1d(signal_1d, x=x_1d, method='chebyshev', deg_num=25, deg_den=25)
    restored_1d = decompress_signal_1d(frac_1d, inter_1d, N)

    plot_signal_and_approx(x_1d, signal_1d, restored_1d, title="Сжатие 1D (Chebyshev, deg=25/25)")
    err_1d = compute_basic_metrics(signal_1d, restored_1d)
    print("Ошибки 1D сжатия:", err_1d)

    # === 4. 2D-сжатие изображения (Bernstein) ===
    print("\n=== 4) 2D-сжатие изображения ===")
    H, W = 80, 60
    cx, cy = (H - 1) / 2, (W - 1) / 2
    image_rgb = np.zeros((H, W, 3), dtype=float)

    for i in range(H):
        for j in range(W):
            dist = math.hypot(i - cx, j - cy) / math.hypot(cx, cy)
            lin = i / (H - 1)
            diag = math.sin(2 * math.pi * (i + j) / (H + W))
            wave = 0.5 * math.sin(2 * math.pi * i / 20) + 0.3 * math.cos(2 * math.pi * j / 15)

            image_rgb[i, j, 0] = dist + 0.05 * (np.random.rand() - 0.5)
            image_rgb[i, j, 1] = lin + 0.3 * diag
            image_rgb[i, j, 2] = wave

    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())

    # Сжатие и восстановление поканально
    fractions_rgb = []
    for c in range(3):
        fracs, size = compress_image_2d(image_rgb[:, :, c], method='bernstein', deg_num=20, deg_den=20)
        fractions_rgb.append(fracs)

    restored_rgb = np.zeros_like(image_rgb)
    for c in range(3):
        restored_rgb[:, :, c] = decompress_image_2d(fractions_rgb[c], size)

    # Визуализация
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Оригинал")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(restored_rgb)
    plt.title("Восстановление")
    plt.axis('off')
    plt.suptitle("2D-сжатие (Bernstein)")
    plt.show()

    # Ошибки поканально
    for c, name in zip(range(3), ['R', 'G', 'B']):
        mse = np.mean((image_rgb[:, :, c] - restored_rgb[:, :, c]) ** 2)
        print(f"MSE канал {name}: {mse:.2e}")

    # === 5. Проверка ускорения на heavy_func ===
    print("\n=== 5) Аппроксимация heavy_func (Bernstein) ===")

    def approximate_heavy(parallel: bool):
        return BernsteinFraction.approximate(
            func=heavy_func,
            interval=(0, 1),
            deg_num=5, deg_den=5,
            poles=None, max_iter=10, tol=1e-8, parallel=parallel
        )

    frac_no_par, t_no_par = measure_time(approximate_heavy, False)
    print(f"Время (без параллели): {t_no_par:.3f} сек")

    frac_par, t_par = measure_time(approximate_heavy, True)
    print(f"Время (с параллелью):   {t_par:.3f} сек")

    x_test = np.linspace(0, 1, 50)
    y_true = np.array([heavy_func(x) for x in x_test])
    y_np = frac_no_par.evaluate(x_test)
    y_pp = frac_par.evaluate(x_test)

    print(f"MSE без параллели: {np.mean((y_true - y_np) ** 2):.2e}")
    print(f"MSE с параллелью:  {np.mean((y_true - y_pp) ** 2):.2e}")

    print("\n=== Демонстрация завершена ===")


if __name__ == "__main__":
    main()
