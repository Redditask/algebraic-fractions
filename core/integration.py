"""
Здесь можем разместить функции, которые напрямую "интегрируют" дроби,
используя их метод evaluate(x).
"""

from core.calculation import numerical_integration

def integrate_fraction(fraction, a, b, n=1000):
    """
    Интегрирует fraction.evaluate(x) на отрезке [a, b],
    применяя метод numerical_integration (метод трапеций).

    :param fraction: объект, имеющий метод evaluate(x),
                     например ChebyshevMarkovFraction или BernsteinFraction
    :param a: float
    :param b: float
    :param n: int
    :return: float, оценка \int_a^b fraction.evaluate(x) dx
    """
    def local_func(x_array):
        return fraction.evaluate(x_array)
    return numerical_integration(local_func, a, b, n=n)
