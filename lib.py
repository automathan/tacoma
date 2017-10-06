import itertools
import matplotlib.pyplot as plt

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
# The following function is not directly related to the project
# but rather convenience functions to make the rest of the code more readable
# it plots most of our functions.


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


def secant_method(f, x_1, x_2, tolerance=0.5e-3, y=0):
    """
    We use the secant method for finding roots when the
    derivative is unknown (which is the case in this project (when considering the higher order functions))
    """
    x_previous = x_1
    x_current = x_2
    f_current = f(x_previous) - y

    # Only stop iterating when the x values are close enough to each other
    while np.abs(x_previous - x_current) > tolerance:
        # Calculate the new f(x) value and swap the old one into the previous f
        f_current, f_previous = f(x_current) - y, f_current

        # Set the current x according to the secant method, and at the same time swap in current x into previous
        x_previous, x_current = x_current, x_current - f_current * (x_current - x_previous) / (f_current - f_previous)

    return x_current

#
# ODE, Euler
#


# The following 4 functions are functions that return the step function for the given method,
# these are used in the euler function below(passed in as the method parameter).

def euler_normal(f, h, t):
    return lambda w, i: w + h * f(t(i-1), w)


def euler_trapezoid(f, h, t):
    def trapezoid_step(w, i):
        t_i = t(i-1)
        w_n = f(t_i, w)
        return w + h*(w_n + f(t_i+h, w + h*w_n))/2

    return trapezoid_step


def euler_midpoint(f, h, t):
    def midpoint_step(w, i):
        t_i = t(i-1)
        w_n = f(t_i, w)
        return w + h*f(t_i+h/2, w + h*w_n/2)

    return midpoint_step


def euler_rk4(f, h, t):
    def rk4_step(w, i):
        t_i = t(i-1)

        s1 = f(t_i, w)
        s2 = f(t_i + (h / 2), w + (h / 2) * s1)
        s3 = f(t_i + (h / 2), w + (h / 2) * s2)
        s4 = f(t_i + h, w + h * s3)
        return w + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4)

    return rk4_step


def euler(f, h=1, t=None, iv=None, method=euler_normal):
    fro, to = iv[0], t

    n: float = (to - fro) / h
    if not (n.is_integer() and n > 0):
        raise Exception("Number of iterations must be a positive integer.")

    n = int(n)

    t_i = lambda i: fro + i/(1/h)  # Trying to avoid floating point rounding errors
    step = method(f, h, t_i)
    w = iv[1]
    yield dict(i=0, t=t_i(0), w=w)

    for i in range(1, n+1):
        w = step(w, i)

        yield dict(i=i, t=t_i(i), w=w)


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


def scalar(y):
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

    def step_generator(f, to_scalar=scalar):
        s = [0.0 for _ in s_coefficients]

        # Define the actual variable step function
        def step(h, t, y):
            # Define a helper function that captures a very common operation in all rk methods
            # namely summing up each value in the input vector interleaved with
            # the s vector. Mathematically speaking we sum
            # the set [h * c * s_i | (s_i, c) ∈ interleave(s, coefficients)]
            def slope(coefficients):
                return sum(h * c * s_i for s_i, c in zip(s, coefficients))

            for i in range(len(s)):
                s[i] = f(t + s_coefficients[i] * h, y + slope(tableau[i]))

            a_w = slope(orders_coefficients[1])
            a_z = slope(orders_coefficients[0])

            # return a tuple of (5th order z, error estimate, 4th order w)
            return y + a_z, np.abs(to_scalar(a_w - a_z)), y + a_w

        return step, len(s) - 1
    return step_generator


def variable_euler(f, t=None, iv=None, method=rkf45(), tolerance=1e-14, start_h=0.1, safety_factor=0.8, to_scalar=scalar, fps=False):
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


if __name__ == '__main__':
    f = lambda t, y: t*y+t**3

    variable_euler(f, t=1, iv=(0, 1), tolerance=1e-10)
