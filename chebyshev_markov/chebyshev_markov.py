"""
Реализует класс ChebyshevMarkovFraction, который:
 - хранит коэффициенты числителя и знаменателя (полиномы Чебышева T или U),
   плюс полюса (при необходимости),
 - предоставляет метод evaluate(x),
 - содержит статический метод approximate() для итерационной аппроксимации
   (вызов функции iterative_chebyshev_rational_approx из core.approximation).
"""

import numpy as np
from numpy.polynomial import chebyshev as cheb

# Импортируем наш итерационный метод
from core.approximation import iterative_chebyshev_rational_approx

class ChebyshevMarkovFraction:
    """
    Класс, описывающий алгебраическую дробь Чебышева–Маркова:
      R(x) = Num(x)/Den(x),
    где Num(x) и Den(x) — линейные комбинации полиномов Чебышева (T_k или U_k),
    а также могут учитываться полюса.
    """

    def __init__(self, num_coeffs, den_coeffs, poles=None, kind='T'):
        """
        :param num_coeffs: np.ndarray
            Коэффициенты числителя (Chebyshev). Размер deg_num+1.
        :param den_coeffs: np.ndarray
            Коэффициенты знаменателя (Chebyshev). Размер deg_den+1.
        :param poles: list or np.ndarray
            Полюса (x - p_i) для знаменателя.
        :param kind: 'T' или 'U'
            Какой набор полиномов (T_k, U_k).
        """
        self.num_coeffs = np.array(num_coeffs, dtype=float)
        self.den_coeffs = np.array(den_coeffs, dtype=float)
        self.poles = np.array(poles) if poles is not None else np.array([])
        self.kind = kind

    def evaluate(self, x):
        """
        Вычисляет R(x) = Num(x)/Den(x), где Num, Den ~ Chebyshev.
        Учёт полюсов: Den(x) *= \prod (x - p_i).

        :param x: float или np.ndarray
        :return: float или np.ndarray
        """
        x_arr = np.array(x, ndmin=1, dtype=float)

        if self.kind == 'T':
            num_vals = cheb.chebval(x_arr, self.num_coeffs)
            den_vals = cheb.chebval(x_arr, self.den_coeffs)
        else:
            # kind='U'
            num_vals = cheb.chebval(x_arr, self.num_coeffs, kind='u')
            den_vals = cheb.chebval(x_arr, self.den_coeffs, kind='u')

        for p in self.poles:
            den_vals *= (x_arr - p)

        eps = 1e-15
        den_vals = np.where(np.abs(den_vals)<eps, eps, den_vals)

        result = num_vals / den_vals
        return result if result.size>1 else float(result[0])

    @staticmethod
    def approximate(
        func,
        interval=(-1,1),
        deg_num=3,
        deg_den=3,
        kind='T',
        poles=None,
        max_iter=5,
        tol=1e-12,
        parallel=False
    ):
        """
        Статический метод, создаёт ChebyshevMarkovFraction,
        используя iterative_chebyshev_rational_approx из core.approximation.

        :param func: callable
        :param interval: (float,float)
        :param deg_num: int
        :param deg_den: int
        :param kind: 'T' or 'U'
        :param poles: list of float
        :param max_iter: int
        :param tol: float
        :param parallel: bool
        :return: ChebyshevMarkovFraction
        """
        num_coeffs, den_coeffs = iterative_chebyshev_rational_approx(
            func=func,
            interval=interval,
            deg_num=deg_num,
            deg_den=deg_den,
            kind=kind,
            poles=poles,
            max_iter=max_iter,
            tol=tol,
            parallel=parallel
        )
        return ChebyshevMarkovFraction(num_coeffs, den_coeffs, poles, kind)
