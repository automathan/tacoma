import time
import itertools
from queue import Queue
from threading import Thread


def consume(results, que):
    for result in results:
        que.put(result)


def smooth_time(results, t_key='t', playback_speed=1.0, rate=None):
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
            time_elapsed_since_yield = time.perf_counter() - last_yield

        time.sleep(max(dt - time_elapsed_since_yield, 0))

        r = que.get(timeout=dt)
        last_yield = time.perf_counter()
        yield r

    t.join()


if __name__ == '__main__':
    import lib

    s = time.perf_counter()

    for result in smooth_time(lib.euler(lambda t, y: t, h=0.1, t=1, iv=(0, 1))):
        print(f"Time: {time.perf_counter() - s}")
        print(result)
