import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections  as mc

# Create new Figure with black background
fig = plt.figure(figsize=(8, 8), facecolor='black')

# Add a subplot with no frame
ax = plt.subplot(111, frameon=False)

# No ticks
ax.set_xticks([])
ax.set_yticks([])

ax.set_ylim(-100, 100)
ax.set_xlim(-100, 100)

t = 0
increment = 0.001
a = [-40, 0]
b = [40, 0]
a_t = [40, 30]
b_t = [-40, 30]

line, = ax.plot(a, b)
left, = ax.plot(a, a_t)
right, = ax.plot(b, b_t)


def points(y, theta):
    a = (40 * np.cos(theta), 40 * np.sin(theta) - y)
    b = (-40 * np.cos(theta), -40 * np.sin(theta) - y)

    line.set_data(*zip(a, b))
    left.set_data(*zip(a, a_t))
    right.set_data(*zip(b, b_t))

    return left, line, right


if __name__ == '__main__':
    from timing import steady_time

    import lib

    d, K, m, a, W, l, omega = 0.01, 1000, 2500, 0.2, 0, 6, 2 * np.pi * (38 / 60)

    iv = np.matrix([
        [1],
        [0],
        [0.001],
        [0]
    ])

    b = K / (m * a)

    def f(t, y):
        s = np.sin(y[2, 0])
        a1 = np.exp(a * (y[0, 0] - l * s))
        a2 = np.exp(a * (y[0, 0] + l * s))

        return np.matrix([
            [y[1, 0]],
            [-d * y[1, 0] - b * (a1 + a2 - 2) + 0.2 * W * np.sin(omega * t)],
            [y[3, 0]],
            [-d * y[3, 0] + b * (3 * np.cos(y[2, 0]) / l) * (a1 - a2)]
        ])

    def update(fr, *args):
        y = fr['w']
        return points(y[0, 0], y[2, 0])

    anim = animation.FuncAnimation(fig, update, frames=steady_time(lib.euler(f, h=0.1, t=100, iv=(0, iv)), playback_speed=20), interval=1 / 10)
    plt.show()