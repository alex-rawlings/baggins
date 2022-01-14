import time
import multiprocessing as mp

def cube(x):
    current = mp.current_process()
    print("Hello from {}".format(current.pid))
    return x*x*x

if __name__ == "__main__":
    print("Starting")
    ts = time.time()
    with mp.Pool(processes=4) as pool:
        res = pool.map(cube, range(100))
        #print(res)
        pool.close()
        pool.join()
    print("Time: {:.3f}".format(time.time() - ts))