from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "r").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)


class Bridge:
    def __init__(self, l):
        # Add a subplot with no frame
        ax = plt.subplot(111, frameon=False)
        self.len = l*12

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_ylim(-100, 100)
        ax.set_xlim(-100, 100)

        a = [0, 0]
        b = [0, 0]
        self.a_t = [self.len, 30]
        self.b_t = [-self.len, 30]

        self.line, = ax.plot(a, b)
        self.left, = ax.plot(a, self.a_t)
        self.right, = ax.plot(b, self.b_t)

    def lines(self, y, theta):
        a = (self.len * np.cos(theta), self.len * np.sin(theta) - y)
        b = (-self.len * np.cos(theta), -self.len * np.sin(theta) - y)

        self.line.set_data(*zip(a, b))
        self.left.set_data(*zip(a, self.a_t))
        self.right.set_data(*zip(b, self.b_t))

        return self.left, self.line, self.right


if __name__ == '__main__':
    from timing import smooth_time

    import lib
    import problems

    iv = np.matrix([
        [0],
        [0],
        [0.001],
        [0]
    ])

    l = 6

    # Create new Figure with black background
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    br = Bridge(l)

    i = 0
    def update(fr, *args):
        global i
        i += 1
        y = fr['w']
        # print(fr['w'][0, 0])
        return br.lines(y[0, 0], y[2, 0])


    # 78.96
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter()
    results = lib.euler(problems.tacoma(W=80, l=l), t=1000, h=1/24, iv=(0, iv), method=lib.euler_rk4)
    frame_count = 1000*24
    anim = animation.FuncAnimation(fig, update, frames=results, interval=100/frame_count, save_count=int(frame_count))
    anim.save('bridge.mp4', writer=writer)
    print(i)

    # import time
    #
    # start = time.perf_counter()
    # for result in results:
    #     pass
    #
    # time_taken = time.perf_counter() - start
    #
    # print(time_taken)
    # print(1000 / time_taken)
    # print("animation finished sucessfully")