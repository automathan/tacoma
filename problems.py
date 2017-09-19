import numpy as np


def tacoma(l=6, m=2500, d=0.01, omega=2 * np.pi * (38 / 60), W=0):
    a = 0.2
    c = 1000 / (m * a)

    def f(t, y):
        s = np.sin(y[2, 0])
        a1 = np.exp(a * (y[0, 0] - l * s))
        a2 = np.exp(a * (y[0, 0] + l * s))

        return np.matrix([
            [y[1, 0]],
            [-d * y[1, 0] - c * (a1 + a2 - 2) + 0.2 * W * np.sin(omega * t)],
            [y[3, 0]],
            [-d * y[3, 0] + c * (3 * np.cos(y[2, 0]) / l) * (a1 - a2)]
        ])

    return f