import os


__all__ = ["get_cpu_count"]


def get_cpu_count():
    return len(os.sched_getaffinity(0))