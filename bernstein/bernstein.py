"""
bernstein.py

Реализует класс BernsteinFraction, описывающий алгебраическую дробь
в 'бернштейновском' базисе x^k:
  R(x) = Num(x)/Den(x),
где Num(x) = \sum_{i=0}^{deg_num} a_i x^i,
    Den(x) = 1 + \sum_{j=1}^{deg_den} b_j x^j,
    + (x - p_i) для учёта полюсов (при необходимости).

Для аппроксимации используем iterative_bernstein_rational_approx из core.approximation,
который возвращает единый массив coeffs длины (deg_num+1 + deg_den).
"""

import numpy as np
from core.approximation import iterative_bernstein_rational_approx

class BernsteinFraction:
    """
    Алгебраическая дробь Бернштейна:
      R(x) = Num(x)/Den(x),
    где Num(x) = a0 + a1*x + ... + a_deg_num*x^deg_num,
        Den(x) = 1 + b1*x + ... + b_deg_den*x^deg_den
        и, при необходимости, полюса (x - p_i).
    """

    def __init__(self, coeffs, deg_num, deg_den, poles=None):
        """
        :param coeffs: np.ndarray
            Массив длины (deg_num+1 + deg_den),
            где первые (deg_num+1) коэффициентов - это a_0..a_deg_num (числитель),
            а следующие (deg_den) - b_1..b_deg_den (знаменатель без начальной '1').
        :param deg_num: int
        :param deg_den: int
        :param poles: list or np.ndarray
            Полюса, умножаемые на знаменатель (x - p_i).
        """
        self.deg_num = deg_num
        self.deg_den = deg_den
        # Делим массив coeffs по docstring:
        self.num_coeffs = np.array(coeffs[:deg_num+1], dtype=float)   # a_0..a_deg_num
        self.den_coeffs = np.array(coeffs[deg_num+1:], dtype=float)   # b_1..b_deg_den
        self.poles = np.array(poles) if poles is not None else np.array([])

    def evaluate(self, x):
        """
        Вычисляет значение дроби в точке(ах) x.

        Num(x) = \sum_{i=0..deg_num} a_i x^i.
        Den(x) = 1 + \sum_{j=1..deg_den} b_j x^j,
                 затем домножается на \prod (x - p_i) (если poles есть).
        """
        x_arr = np.array(x, ndmin=1, dtype=float)

        # 1) Числитель
        num_vals = np.zeros_like(x_arr)
        for i, ai in enumerate(self.num_coeffs):
            num_vals += ai * (x_arr**i)

        # 2) Знаменатель
        den_vals = np.ones_like(x_arr)
        for j, bj in enumerate(self.den_coeffs, start=1):
            den_vals += bj * (x_arr**j)

        # Полюса
        for p in self.poles:
            den_vals *= (x_arr - p)

        # Защита от деления на 0
        eps = 1e-15
        den_vals = np.where(np.abs(den_vals) < eps, eps, den_vals)

        result = num_vals / den_vals
        return result if result.size>1 else float(result[0])

    @staticmethod
    def approximate(
        func,
        interval=(0,1),
        deg_num=3,
        deg_den=3,
        poles=None,
        max_iter=5,
        tol=1e-12,
        parallel=False
    ):
        """
        Строит дробь Бернштейна, вызывая iterative_bernstein_rational_approx.

        :param func: callable
        :param interval: (float,float)
        :param deg_num: int
        :param deg_den: int
        :param poles: list or None
        :param max_iter: int
        :param tol: float
        :param parallel: bool
        :return: BernsteinFraction
        """
        coeffs = iterative_bernstein_rational_approx(
            func=func,
            interval=interval,
            deg_num=deg_num,
            deg_den=deg_den,
            poles=poles,
            max_iter=max_iter,
            tol=tol,
            parallel=parallel
        )
        # Теперь coeffs - массив длины (deg_num+1 + deg_den)
        return BernsteinFraction(coeffs, deg_num, deg_den, poles=poles)
