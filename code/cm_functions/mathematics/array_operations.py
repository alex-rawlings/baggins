import numpy as np


__all__ = ["get_histogram_bin_centres"]


def get_histogram_bin_centres(bins):
    """
    Convenience function to get the centres of some histogram bins.

    Parameters
    ----------
    bins : np.ndarray
        bin edges

    Returns
    -------
    : np.ndarray
        bin centres, has len = len(bins)-1
    """
    return (bins[:-1] + bins[1:]) / 2
