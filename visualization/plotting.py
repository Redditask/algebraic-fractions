"""
Универсальные функции построения графиков,
например, сравнение оригинального массива и восстановленного
(например, при сжатии).
"""

import matplotlib.pyplot as plt

def plot_signal_and_approx(
    x,
    signal,
    approx_signal,
    title="Сравнение сигнала и аппроксимации"
):
    """
    Строит один график: original vs approximated.

    :param x: np.ndarray
    :param signal: np.ndarray (original)
    :param approx_signal: np.ndarray (approximated)
    :param title: str
    """
    plt.figure()
    plt.plot(x, signal, 'o-', label='Оригинальный сигнал')
    plt.plot(x, approx_signal, 'x--', label='Аппроксимация')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_2d_data(original_2d, restored_2d, title="Сравнение 2D-данных"):
    """
    Сравнение двух 2D-массивов (например, картинки).
    """
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(original_2d, cmap='gray')
    plt.title("Оригинал")

    plt.subplot(1,2,2)
    plt.imshow(restored_2d, cmap='gray')
    plt.title("Восстановлено")

    plt.suptitle(title)
    plt.show()
