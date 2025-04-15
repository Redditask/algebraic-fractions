"""
Здесь лежат итерационные методы аппроксимации для:
 1) Чебышева–Маркова (полиномы Чебышева T/U + полюса)
 2) Бернштейна (степенные полиномы + полюса)

Мы можем задействовать:
 - core.calculation.solve_least_squares (задача МНК)
 - utils.helpers.generate_points (генерировать сетку точек)
 - core.parallel_acceleration.polynomial_powers_parallel (для ускорения построения x^k)
 - joblib.Parallel (для параллельного вычисления func(x) и построения матриц)
"""

import numpy as np
from utils.helpers import generate_points
from core.calculation import solve_least_squares
from core.parallel_acceleration import polynomial_powers_parallel
from numpy.polynomial import chebyshev as cheb


def iterative_chebyshev_rational_approx(
    func,
    interval,
    deg_num,
    deg_den,
    kind='T',
    poles=None,
    max_iter=5,
    tol=1e-12,
    parallel=False
):
    """
    Итерационная аппроксимация функции func на заданном интервале,
    используя полиномы Чебышева (T_k или U_k) и, при необходимости, полюса.

    Подразумеваем рациональную дробь вида:
       R(x) = Num(x)/Den(x),
    где Num(x) = \sum_{k=0}^{deg_num} alpha_k * T_k(x),
        Den(x) = \sum_{j=0}^{deg_den} beta_j * T_j(x)
                 (после чего учитываем полюса: умножаем Den(x) на (x - p_i)).

    Алгоритм итерационный (упрощённый аналог):
    1) Инициализируем (alpha, beta) ~ (0..0), beta_0=1
    2) На каждом шаге:
       - Den_i(x) = chebval(x, beta) * \prod (x - p_i)
       - Num_i(x) обновляем, решая y * Den_i(x) ~ Num_i(x) по МНК
       - Den_{i+1}(x) обновляем, решая Num_i(x)/y ~ Den_{i+1}(x), тоже по МНК
       - Проверяем сходимость по норме изменения (alpha, beta)

    :param func: callable
        Функция, которую аппроксимируем (func: float->float).
    :param interval: (float, float)
        Интервал [xmin, xmax], на котором строим аппроксимацию.
    :param deg_num: int
        Степень числителя (количество полиномиальных коэффициентов = deg_num+1).
    :param deg_den: int
        Степень знаменателя (количество полиномиальных коэффициентов = deg_den+1).
    :param kind: 'T' или 'U'
        Какой тип полиномов Чебышева использовать (T_k или U_k).
        'T' часто соответствует косинус-подходу, 'U' ~ синус-подход.
    :param poles: list of float
        Полюса (каждый p => (x - p)) домножаем к знаменателю,
        придавая дроби особую форму.
    :param max_iter: int
        Максимальное число итераций для алгоритма.
    :param tol: float
        Порог сходимости (по Евклидовой норме изменений коэф.).
    :param parallel: bool
        Если True, параллелим вычисление y_points = func(x) и построение матриц
        с помощью joblib.Parallel.
    :return: (num_coeffs, den_coeffs) -- np.ndarray, np.ndarray
        Наборы коэффициентов (alpha, beta) для числителя и знаменателя.

    Пример использования:
        alpha, beta = iterative_chebyshev_rational_approx(
            func=np.sin, interval=(-1,1), deg_num=5, deg_den=5, parallel=True
        )
    """

    if poles is None:
        poles = []

    # Шаг 1: генерируем сетку x_points
    x_points = generate_points(interval, num=200)

    # При parallel=True, параллелим вычисление func(x) через joblib
    if parallel:
        from joblib import Parallel, delayed
        y_vals = Parallel(n_jobs=-1)(delayed(func)(xi) for xi in x_points)
        y_points = np.array(y_vals, dtype=float)
    else:
        y_points = np.array([func(xi) for xi in x_points], dtype=float)

    # Инициализация коэффициентов
    num_coeffs = np.zeros(deg_num+1, dtype=float)
    den_coeffs = np.zeros(deg_den+1, dtype=float)
    den_coeffs[0] = 1.0  # пусть Den(x) начинается с 1 + ...

    old_num = num_coeffs.copy()
    old_den = den_coeffs.copy()

    for iteration in range(max_iter):
        # 1) Den_i(x) = chebval(x_points, den_coeffs), * полюса
        if kind == 'T':
            den_raw = cheb.chebval(x_points, den_coeffs)
        else:
            den_raw = cheb.chebval(x_points, den_coeffs, kind='u')
        for p in poles:
            den_raw *= (x_points - p)

        # 2) Обновляем num_coeffs, решая:
        #    y_points * den_raw ~ Num(x)
        #    Num(x) = \sum_{k=0}^{deg_num} alpha_k * T_k(x)
        if parallel:
            from joblib import Parallel, delayed

            def single_col(k):
                return cheb.Chebyshev.basis(k)(x_points)

            columns = Parallel(n_jobs=-1)(delayed(single_col)(k) for k in range(deg_num+1))
            A_num = np.column_stack(columns)
        else:
            A_num = np.zeros((len(x_points), deg_num+1), dtype=float)
            for k in range(deg_num+1):
                A_num[:, k] = cheb.Chebyshev.basis(k)(x_points)

        b_num = y_points * den_raw
        alpha = solve_least_squares(A_num, b_num)
        num_coeffs = alpha

        # 3) Обновляем den_coeffs, решая:
        #    num_raw(x) ~ y_points * Den(x)
        #    Den(x) = \sum_{j=0}^{deg_den} beta_j * T_j(x)
        if kind == 'T':
            num_raw = cheb.chebval(x_points, num_coeffs)
        else:
            num_raw = cheb.chebval(x_points, num_coeffs, kind='u')

        if parallel:
            def single_col_den(j):
                return cheb.Chebyshev.basis(j)(x_points)
            cols_den = Parallel(n_jobs=-1)(delayed(single_col_den)(j) for j in range(deg_den+1))
            A_den = np.column_stack(cols_den)
        else:
            A_den = np.zeros((len(x_points), deg_den+1), dtype=float)
            for j in range(deg_den+1):
                A_den[:, j] = cheb.Chebyshev.basis(j)(x_points)

        mask = np.abs(y_points) > 1e-14
        b_den = np.zeros_like(num_raw)
        b_den[mask] = num_raw[mask] / y_points[mask]

        beta = solve_least_squares(A_den[mask, :], b_den[mask])
        den_coeffs = beta

        # 4) Критерий сходимости
        diff_num = np.linalg.norm(num_coeffs - old_num)
        diff_den = np.linalg.norm(den_coeffs - old_den)
        if diff_num < tol and diff_den < tol:
            break

        old_num = num_coeffs.copy()
        old_den = den_coeffs.copy()

    return num_coeffs, den_coeffs


def iterative_bernstein_rational_approx(
    func,
    interval,
    deg_num,
    deg_den,
    poles=None,
    max_iter=5,
    tol=1e-12,
    parallel=False
):
    """
    Итерационная аппроксимация func на [interval],
    используя "бернштейновский" базис x^k (степенные полиномы) + полюса.

    Рассматриваем дробь вида:
        R(x) = Num(x) / Den(x),
    где:
        - Num(x) = a_0 + a_1 * x + ... + a_deg_num * x^deg_num,
        - Den(x) = 1 + b_1 * x + ... + b_deg_den * x^deg_den,
          причём домножаем Den(x) на (x - p_i) для каждого полюса p_i (если есть).

    Алгоритм (упрощённый, итерационный):
    1) Генерируем сетку x_points (и y_points = func(x_points)).
    2) den(x) = 1 + sum_{j=1..deg_den} b_j x^j; при этом умножаем на \prod (x - p_i).
    3) num(x) ~ y_points * den(x) => находим a_i (числитель).
    4) den(x) ~ num(x)/y_points => находим b_j (знаменатель).
    5) Проверяем сходимость (по norm изменения коэффициентов).

    :param func: callable
    :param interval: (float,float)
    :param deg_num: int
        Степень числителя (число коэффициентов = deg_num+1).
    :param deg_den: int
        Степень знаменателя (число коэффициентов = deg_den).
        В итоге массив коэффициентов будет иметь длину (deg_num+1) + deg_den.
    :param poles: list of float
    :param max_iter: int
    :param tol: float
    :param parallel: bool
        При True используем joblib и polynomial_powers_parallel для ускорения
        (распараллеливаем вычисление func(x) и построение x^k).
    :return:
        coeffs : np.ndarray длины (deg_num+1 + deg_den),
        где первые (deg_num+1) коэффициентов относятся к числителю (a_0..a_deg_num),
        последние deg_den коэффициентов относятся к знаменателю (b_1..b_deg_den).

    Пример:
        coeffs = iterative_bernstein_rational_approx(
            func=np.sin, interval=(0, np.pi),
            deg_num=4, deg_den=4,
            parallel=True
        )
        # => Класс BernsteinFraction конструктором разрежет coeffs на num/den.
    """
    if poles is None:
        poles = []

    # Генерируем x_points, y_points
    x_points = generate_points(interval, num=200)

    if parallel:
        from joblib import Parallel, delayed
        y_vals = Parallel(n_jobs=-1)(delayed(func)(xi) for xi in x_points)
        y_points = np.array(y_vals, dtype=float)
    else:
        y_points = np.array([func(x) for x in x_points], dtype=float)

    # Всего coeffs = (deg_num+1) + deg_den
    total_len = (deg_num + 1) + deg_den
    coeffs = np.zeros(total_len, dtype=float)
    old_coeffs = coeffs.copy()

    for it in range(max_iter):
        # Разбиваем coeffs на a (числитель), b (знаменатель)
        half = deg_num + 1
        a = coeffs[:half]    # a_0 .. a_deg_num
        b = coeffs[half:]    # b_1 .. b_deg_den

        # den_raw(x) = (1 + \sum_j b_j x^j) * \prod_p (x - p)
        den_raw = np.ones_like(x_points)
        for j, bj in enumerate(b, start=1):
            den_raw += bj * (x_points**j)
        for p in poles:
            den_raw *= (x_points - p)

        # num(x) ~ y_points * den_raw(x)
        if parallel:
            A_num = polynomial_powers_parallel(x_points, deg_num)  # shape = (N, deg_num+1)
        else:
            A_num = np.column_stack([x_points**k for k in range(deg_num+1)])

        b_num = y_points * den_raw
        a_new = solve_least_squares(A_num, b_num)

        # den(x) ~ num(x)/y_points
        num_raw = np.zeros_like(x_points)
        for i, ai in enumerate(a_new):
            num_raw += ai * (x_points**i)

        # b_den_vals = num_raw / y_points
        mask = np.abs(y_points) > 1e-14
        b_den_vals = np.zeros_like(num_raw)
        b_den_vals[mask] = num_raw[mask]/ y_points[mask]

        # Строим матрицу A_den => x^1..x^deg_den
        if parallel:
            A_den_full = polynomial_powers_parallel(x_points, deg_den)
            A_den = A_den_full[:, 1:]  # выкидываем x^0
        else:
            A_den = np.column_stack([x_points**j for j in range(1, deg_den+1)])

        b_new = solve_least_squares(A_den[mask,:], b_den_vals[mask])

        new_coeffs = np.concatenate([a_new, b_new])
        # проверяем сходимость
        if np.linalg.norm(new_coeffs - old_coeffs) < tol:
            coeffs = new_coeffs
            break

        coeffs = new_coeffs
        old_coeffs = new_coeffs.copy()

    return coeffs
