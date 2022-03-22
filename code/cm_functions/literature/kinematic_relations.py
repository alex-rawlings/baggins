import numpy as np

_all__ = ["Kratsov13"]

def Kratsov13(Rvir):
    """
    Define the size-virial relation from Kravtsov 2012
    https://iopscience.iop.org/article/10.1088/2041-8205/764/2/L31/pdf

    Parameters
    ----------
    Rvir: virial radius in kpc

    Returns
    -------
    half mass radius [kpc]
    """
    return (0.015*Rvir)