import os
import shutil
import atexit
from ..env_config import _cmlogger, TMPDIRs


_logger = _cmlogger.getChild(__name__)


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
    Delete any user defined temporary directories, if created.
    """
    for d in TMPDIRs.register:
        # delete directory
        shutil.rmtree(d)
        _logger.warning(f"Deleted temporary directory {d}")
