import numpy as np

"""
A collection of miscellaneous relations that are somehow dependent on a radial
value. 
"""
__all__ = ['Sahu20']


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
