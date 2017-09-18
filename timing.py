import time
import numpy as np
import itertools
from queue import Queue
from threading import Thread


def consume(results, que):
    for result in results:
        que.put(result)


def steady_time(results, t_key='t', playback_speed=1.0, rate=None):
    if rate is None:
        g = iter(results)
        probe = list(itertools.islice(g, 2))

        t1, t2 = probe[0][t_key], probe[1][t_key]
        dt = (t2 - t1) / playback_speed

        results_generator = itertools.chain(probe, g)
    else:
        dt = 1 / (rate * playback_speed)
        results_generator = iter(results)

    que = Queue()
    t = Thread(target=consume, args=(results_generator, que))
    t.start()

    last_yield = None

    while t.is_alive() or not que.empty():
        if not last_yield:
            time_elapsed_since_yield = dt
        else:
            time_elapsed_since_yield = time.process_time() - last_yield

        time.sleep(max(dt - time_elapsed_since_yield, 0))

        r = que.get(timeout=dt)
        last_yield = time.process_time()
        yield r

    t.join()


if __name__ == '__main__':
    import lib

    d, K, m, a, W, l, omega = 0.01, 1000, 2500, 0.2, 80, 6, 2*np.pi*(38/60)

    iv = np.matrix([
        [0],
        [0],
        [0.001],
        [0]
    ])

    b = K/m*a


    def f(t, y):
        s = np.sin(y[2, 0])
        a1 = np.exp(a*(y[0, 0]-l*s))
        a2 = np.exp(a*(y[0, 0]+l*s))

        return np.matrix([
            [y[1, 0]],
            [-d*y[1, 0]-b*(a1 + a2 - 2)+0.2*W*np.sin(omega*t)],
            [y[3, 0]],
            [-d*y[3, 0]-b*(3*np.cos(y[2, 0])/l)*(a1 + a2)]
        ])

    s = time.perf_counter()

    for result in steady_time(lib.euler(f, h=0.1, t=1, iv=(0, iv))):
        print(f"Time: {time.perf_counter() - s}")
        print(result)
