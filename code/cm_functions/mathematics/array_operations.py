import numpy as np


__all__ = ['get_histogram_bin_centres']


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
