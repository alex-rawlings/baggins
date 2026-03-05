import numpy as np
import h5py
from dask.distributed import Client, as_completed
from dask import delayed
from baggins.env_config import _cmlogger

__all__ = ["any_snapshot_with_single_BH", "get_idx_in_array"]

_logger = _cmlogger.getChild(__name__)


# ---- worker function ----
def _dataset_length(filename):
    with h5py.File(filename, "r") as f:
        return len(f["/PartType5/Masses"])


def any_snapshot_with_single_BH(file_list):
    """
    Determine if any snapshot in a list has only one BH particle. Operation is done in parallel with dask.

    Parameters
    ----------
    file_list : list
        list of a snapshot file names

    Returns
    -------
    single_BH : bool
        is there a snapshot with a single BH?
    """
    client = Client()  # connects to local Dask cluster
    tasks = [delayed(_dataset_length)(fname) for fname in file_list]

    futures = client.compute(tasks)
    single_BH = False
    for future in as_completed(futures):
        length = future.result()
        if length == 1:
            single_BH = True
            # cancel all remaining tasks
            client.cancel(futures)
            break
    client.close()
    return single_BH


def get_idx_in_array(t, tarr):
    """
    Get the index of a value within an array. If multiple matches are found,
    the first is returned (following np.argmin method)

    Parameters
    ----------
    t : int, float
        value to search for
    tarr : array-like
        array to search within

    Returns
    -------
    idx : int
        index of t in tarr

    Raises
    ------
    AssertionError
        value to search for is a nan
    AssertionError
        if index is 0 or the last index of the array
    """
    try:
        assert not np.isnan(t)
    except AssertionError:
        _logger.exception("t must not be nan", exc_info=True)
        raise
    try:
        idx = np.nanargmin(np.abs(tarr - t))
        if idx == len(tarr) - 1:
            s = "large"
            raise AssertionError
        elif idx == 0:
            s = "smalle"
            raise AssertionError
        else:
            return idx
    except AssertionError:
        _logger.exception(f"Value is {s}r than the {s}st array value!", exc_info=True)
        raise
    except ValueError:
        _logger.exception(f"Array tarr has value {np.unique(tarr)}")
        raise
