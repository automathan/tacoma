from functools import wraps
import itertools
from inspect import signature
from IPython.display import display
import matplotlib.pyplot as plt

import sympy as sp
import numpy as np


# - - - Appendix Python - - - 
# In this part we'll highlight some python-specific syntax and functions
# - Indentation is a syntatic construct providing nesting functionality
#
# - "def f(a1, a2=1, ...):" standard function header, where a is for argument, and a2 will have value 1 unless stated otherwise
#
# - "raise" and "assert", used for error handling
# 


# General functions
#
# The first seven following functions are not directly related to the project 
# but rather convenience functions to make the rest of the code more readable

def disallow_none_kwargs(f):
    required_kwargs = []
    for param in signature(f).parameters.values():
        if param.default is None:
            required_kwargs.append(param.name)

    @wraps(f)
    def wrapper(*args, **kwargs):
        for kwarg in required_kwargs:
            if not kwarg in kwargs:
                raise Exception(f"Keyword argument {kwarg} is required.")

        return f(*args, **kwargs)
    return wrapper


def stringify(value):
    if isinstance(value, float):
        return str(value if not value.is_integer() else int(value))
    else:
        return str(value)


def pp_table(table, v_sep='|', h_sep='-', cross_sep='+'):
    just = []
    for key, col in table.items():
        just.append(max(len(stringify(key)), *(len(stringify(cell)) for cell in col)))

    print(f" {v_sep} ".join(header.ljust(just[i]) for i, header in enumerate(table.keys())))
    print(f"{h_sep}{cross_sep}{h_sep}".join(h_sep*just[i] for i, _ in enumerate(table.keys())))

    for row in zip(*table.values()):
        print(f" {v_sep} ".join(stringify(cell).ljust(just[i]) for i, cell in enumerate(row)))


def group_dicts(dicts):
    iterable = iter(dicts)
    head = next(iterable)
    keys = head.keys()

    result = {key: [] for key in keys}
    for key, value in head.items():
        result[key].append(value)

    for dict in iterable:
        assert dict.keys() == keys, "Dictionaries must have same shape"
        for key, value in dict.items():
            result[key].append(value)

    return result


def pp(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        pp_table(group_dicts(fn(*args, **kwargs)))

    return wrapper


def with_error(results, y, x_key='x', y_key='y'):
    for result in results:
        yield {**result, "error": abs(y(result[x_key]) - result[y_key])}


def plot_gen(results, y_keys=None, x_keys=None):
    ys = [[] for _ in y_keys]
    xs = [[] for _ in x_keys]

    for result in results:
        for key, y in zip(y_keys, ys):
            y.append(result[key])

        for key, x in zip(x_keys, xs):
            x.append(result[key])

    for y, x in zip(ys, xs):
        plt.plot(x, y)

#
# Optimization
#

# we use the secant method for finding roots when the derivative is unknown (which is the case in this project (when considering the higher order functions))
def secant_method(f, x_1, x_2, tolerance=0.5e-3, y=0):
    x_previous = x_1
    x_current = x_2
    f_current = f(x_previous) - y

    while np.abs(x_previous - x_current) > tolerance:
        f_current, f_previous = f(x_current) - y, f_current
        x_previous, x_current = x_current, x_current - f_current * (x_current - x_previous) / (f_current - f_previous)

    return x_current

#
# ODE, Euler
#

def ivp(expr, x, ivs):
    eqs = (sp.Eq(expr.subs(x, iv[0]), iv[1]) for iv in ivs)
    free_symbols_solutions = sp.solve(eqs, dict=True)

    if len(free_symbols_solutions) == 0:
        raise Exception(f"Free symbols in expr has no solutions")
    elif len(free_symbols_solutions) > 1:
        raise Exception(f"Free symbols in expr has multiple solutions\n{list(free_symbols_solutions)}")

    return expr.subs(free_symbols_solutions[0])


def euler_normal(f, h, t):
    return lambda w, i: w + h * f(t(i-1), w)


def euler_trapezoid(f, h, t):
    def trapezoid(w, i):
        t_i = t(i-1)
        w_n = f(t_i, w)
        return w + h*(w_n + f(t_i+h, w + h*w_n))/2

    return trapezoid


def euler_midpoint(f, h, t):
    def midpoint(w, i):
        t_i = t(i-1)
        w_n = f(t_i, w)
        return w + h*f(t_i+h/2, w + h*w_n/2)

    return midpoint


def euler_rk4(f, h, t):
    def rk4(w, i):
        t_i = t(i-1)

        s1 = f(t_i, w)
        s2 = f(t_i + (h / 2), w + (h / 2) * s1)
        s3 = f(t_i + (h / 2), w + (h / 2) * s2)
        s4 = f(t_i + h, w + h * s3)
        return w + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4)

    return rk4


def rkf45():
    s_coefficients = np.array([
        0, 1/4, 3/8, 12/13, 1, 1/2
    ])

    # RK45 coefficient matrix

    tableau = [
        [],
        [1 / 4],
        [3 / 32, 9 / 32],
        [1932 / 2197, -7200 / 2197, 7296 / 2197],
        [439 / 216, -8, 3680 / 513, -845 / 4104],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
    ]

    m45 = np.array([
        [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
        [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5]
    ])

    return rk_embedded_pair(s_coefficients, tableau, m45)


def sacalar(y):
    return y if np.isscalar(y) else np.linalg.norm(y, ord=2)


def rk_embedded_pair(s_coefficients, tableau, orders_coefficients):
    for i, row in enumerate(tableau):
        assert len(row) == i, f"The {i}. row in tableau must have an length of {i}"
        assert np.abs(sum(row) - s_coefficients[i]) < 1e-10, f"The coefficients in tableau row {i} must sum to its assosiated s coefficiant."

    for row in orders_coefficients:
        assert np.abs(sum(row) - 1) < 1e-10, "The coefficients in the order methods must sum to 1."

    assert len(orders_coefficients) == 2, "There may only be 2 embedded rk methods."
    assert len(s_coefficients) == len(tableau), "The tableau must have # of rows equal to the length of the s coefficient vector."
    assert len(s_coefficients) == len(orders_coefficients[0]), "The last order method must have the same number of coefficients as the s coefficient vector."
    assert len(orders_coefficients[0]) - 1 == len(orders_coefficients[1]), "The second order method may currently only be one order less than the first."

    def step_generator(f, to_scalar=sacalar):
        s = [0.0 for _ in s_coefficients]

        def step(h, t, y):
            def slope(coefficients):
                return sum(c * h * s_i for s_i, c in zip(s, coefficients))

            for i in range(len(s)):
                s[i] = f(t + s_coefficients[i] * h, y + slope(tableau[i]))

            a_w = slope(orders_coefficients[1])
            a_z = slope(orders_coefficients[0])

            return y + a_z, np.abs(to_scalar(a_w - a_z)), y + a_w

        return step, len(s) - 1
    return step_generator


@disallow_none_kwargs
def variable_euler(f, t=None, iv=None, method=rkf45(), tolerance=1e-14, start_h=0.1, safety_factor=0.8, to_scalar=sacalar, fps=False):
    if fps is not False:
        minh = 1/fps
    else:
        minh = 2**32

    t_goal = t

    def acceptable_error(e, w, magnification=1):
        return e/to_scalar(w) < tolerance/magnification

    step, p = method(f, to_scalar=to_scalar)
    t = iv[0]
    w = iv[1]
    h = start_h

    r = 1/(p+1)
    f = safety_factor * tolerance**r

    yield dict(i=0, t=t, w=w, fourth=w, h=0, error=0)

    _, e, _ = step(h, t, w)
    h = f * h * (to_scalar(w) / max(e, 1e-20)) ** r

    for i in itertools.count(1):
        if t + h >= t_goal - 1e-16:
            h = t_goal - t

            w, e, z = step(h, t, w)
            yield dict(i=i, t=t_goal, w=w, fourth=z, h=h, error=e)
            break

        h = min(h, minh)
        w_next, e, z = step(h, t, w)

        if not acceptable_error(e, w):
            h = f * h * (to_scalar(w) / max(e, 1e-20)) ** r

            w_next, e, z = step(h, t, w)

            while not acceptable_error(e, w):
                h /= 2

                w_next, e, z = step(h, t, w)

        t += h
        w = w_next
        yield dict(i=i, t=t, w=w, fourth=z, h=h, error=e)
        h = f * h * (to_scalar(w) / max(e, 1e-20)) ** r


@disallow_none_kwargs
def euler(f, h=1, t=None, iv=None, method=euler_normal):
    fro, to = iv[0], t

    n = (to - fro) / h  # type: float
    if not (n.is_integer() and n > 0):
        raise Exception("Number of iterations must be a positive integer.")

    n = int(n)

    t_i = lambda i: fro + i/(1/h)  # Trying to avoid floating point rounding errors
    step = method(f, h, t_i)
    w = iv[1]
    yield dict(i=0, t=t_i(0), w=iv[1])

    for i in range(1, n+1):
        w = step(w, i)

        yield dict(i=i, t=t_i(i), w=w)


@disallow_none_kwargs
def euler_error(f, iv=None, multiple_eqs_strategy=lambda eqs: eqs[0], **kwargs):
    y = sp.Function('y')
    t = sp.Symbol('t')

    y_d = sp.Eq(y(t).diff(t), f(t, y(t)))
    diff_eq = sp.dsolve(y_d)

    if isinstance(diff_eq, list):
        diff_eq = multiple_eqs_strategy(diff_eq)

    exact = ivp(diff_eq.rhs, t, [iv])

    display(y_d)
    display(sp.Eq(y(t), exact))

    y_fn = sp.lambdify(t, exact)

    return with_error(euler(f, iv=iv, **kwargs), y_fn, x_key='t', y_key='w')

if __name__ == '__main__':
    f = lambda t, y: t*y+t**3

    pp(variable_euler)(f, t=1, iv=(0, 1), tolerance=1e-10)
    pp(euler_error)(f, h=0.1, t=1, iv=(0, 1), method=euler_rk4)
