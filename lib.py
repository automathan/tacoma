from functools import wraps
from inspect import signature
from IPython.display import display

import sympy as sp

#
# General
#


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


@disallow_none_kwargs
def euler(f, h=1, t=None, iv=None, method=euler_normal):
    fro, to = iv[0], t

    n = (to - fro) / h
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
    f = lambda t, y: t

    pp(euler_error)(f, h=0.1, t=1, iv=(0, 1), method=euler_trapezoid)