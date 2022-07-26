import numpy as np
from dask.distributed import Client, LocalCluster
import dask
import time
from datetime import datetime
import cm_functions as cmf

def myfunc(j, n):
    def _f(i, n):
        time.sleep(2)
        print(i)
        return 3*i+i**2
    jobs = [k for k in range(int(j))]
    '''a = 6
    with LocalCluster(n_workers=4, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            futures = client.map(_f, jobs)
            results = client.gather(futures)'''
    results = []
    for job in jobs:
        y = dask.delayed(_f)(job, n)
        results.append(y)
    results = dask.compute(*results)
    return results


if __name__ == "__main__":
    print(cmf.utils.get_cpu_count())
    now = datetime.now()
    r = myfunc(9, 3)
    print(f"Execution: {datetime.now()-now}")
    print(len(r))
