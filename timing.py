import time
from queue import Queue
from threading import Thread


def consume(results, que):
    for result in results:
        que.put(result)


def smooth_time(results, t_key='t', playback_speed=1.0, buffer=1):
    que = Queue(20)
    t = Thread(target=consume, args=(results, que))
    t.start()

    time.sleep(1)  # Allow que to build up before starting.
    s = time.perf_counter()
    sleep = time.sleep
    perf = time.perf_counter
    get = que.get
    alive = t.is_alive

    while alive():
        next_frame = get()
        dt = next_frame[t_key]/playback_speed - perf() - s
        assert dt + buffer > 0, "Playback is too fast."
        sleep(max(0, dt))

        yield next_frame

    while not que.empty():
        next_frame = get()
        dt = next_frame[t_key] / playback_speed - perf() - s
        assert dt + buffer > 0, "Playback is too fast."
        sleep(max(0, dt))

        yield next_frame

    t.join()


if __name__ == '__main__':
    import lib

    s = time.perf_counter()

    for result in smooth_time(lib.variable_euler(lambda t, y: t, t=10, iv=(0, 1), start_h=0.001)):
        print(f"Time: {time.perf_counter() - s}")
        print(result)
