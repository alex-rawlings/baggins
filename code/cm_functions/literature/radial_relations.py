import numpy as np

"""
A collection of miscellaneous relations that are somehow dependent on a radial
value. 
"""
__all__ = ["OsipkovMerritt", "Sahu20"]


def OsipkovMerritt(r, ra):
    """
    Define the theoretical anisotropy value of an Osipkov-Merritt model with 
    parameter ra

    Parameters
    ----------
    r: array of radial values
    ra: the anisotropy radius

    Returns
    -------
    anisotropy profile beta
    """
    return 1 / (1 + (ra/r)**2)


def Sahu20(logRe):
    """
    Define the Sahu+20 spheroidal mass - Re relation
    Parameters
    ----------
    logRe: log of effective radius in kpc

    Returns
    -------
    expected log spheroidal mass
    """
    return 1.08 * logRe + 10.32
