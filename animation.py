import numpy as np
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


def update(*args):
    global t
    t = (t + increment) % (np.pi/5)

    return points(100*t%20, t)


anim = animation.FuncAnimation(fig, update, interval=1/10)
plt.show()