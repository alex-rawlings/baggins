import os
import shutil
import atexit
from ..env_config import _cmlogger, tmp_dir


_logger = _cmlogger.copy(__file__)


__all__ = ["get_cpu_count"]


def get_cpu_count():
    """
    Get the number of CPUs available.

    Returns
    -------
    : int
        number of CPUs
    """
    return len(os.sched_getaffinity(0))


@atexit.register
def clean_up():
    """
    Delete the user defined temporary directory, if created.
    """
    if os.path.exists(tmp_dir):
        # delete directory
        shutil.rmtree(tmp_dir)
        _logger.logger.warning(f"Deleted temporary directory {tmp_dir}")