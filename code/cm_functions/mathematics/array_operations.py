import numpy as np


__all__ = ['find_index', 'get_histogram_bin_centres']


def find_index(val, arr, tol=1e-7):
    """
    General method to find the index of a given value in a array,
    even if that value is potentially not in the array
    #TODO if val not in arr, ensure entire arr is NOT returned

    Parameters
    ----------
    val: value to find
    arr: array to search
    tol: comparison tolerance: if min(arr-val)< tol -> val in arr

    Returns
    -------
    idx = position of the value closest to val in arr

    Raises
    ------
    AssertionError: val not an int or a float
    RuntimeError: val is not in arr
    """
    assert(isinstance(val, float) or isinstance(val, int))
    #determine which value in arr is closest to val and its position
    mindiff = np.nanmin(np.abs(val-arr))
    #if difference larger than specified tolerance, val isn't really in arr
    if mindiff > tol:
        raise RuntimeError('The value {:.3e} was not found within error {:.3e}!\nThe minimum difference between input <val> and elements in <arr> is {:.3e}'.format(val, tol, mindiff))
    idx = np.where(np.abs(val-arr)-mindiff < tol)[0][0]
    return idx


def get_histogram_bin_centres(bins):
    """
    Convenience function to get the centres of some histogram bins.

    Parameters
    ----------
    bins: array of bin edges

    Returns
    -------
    the bin centres
    """
    return (bins[:-1] + bins[1:]) / 2
