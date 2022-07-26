import time
import multiprocessing as mp

def cube(x):
    current = mp.current_process()
    print("Worker {} received {}".format(current.pid, x))

if __name__ == "__main__":
    import __main__ as main
    print(main.__dict__)
    print("Starting")
    ts = time.time()
    y=2
    with mp.Pool(processes=1) as pool:
        pool.map(cube, range(10))
    print("Time: {:.3f}".format(time.time() - ts))