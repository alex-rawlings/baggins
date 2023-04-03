import numpy as np

__all__ = ["Haring04", "Magorrian98", "Sahu19", "Scott13"]


def Haring04(logMstar):
    """
    define Haring+04 BH-Bulge fit function
    https://ui.adsabs.harvard.edu/abs/2004ApJ...604L..89H/abstract

    Parameters
    ----------
    logMstar : np.ndarray
        logarithm of bulge mass [log(M/Msol)]

    Returns
    -------
    : np.ndarray
        predicted log of bh mass [log(M/Msol)]
    """
    assert(logMstar < 13)
    return 8.20 + 1.12 * (logMstar - 11)


def Magorrian98(logMstar):
    """
    define Magorrian+98 BH-Bulge fit function
    https://ui.adsabs.harvard.edu/abs/1998AJ....115.2285M/abstract

    Parameters
    ----------
    logMstar : np.ndarray
        logarithm of bulge mass [log(M/Msol)]

    Returns
    -------
    : np.ndarray
        predicted log of bh mass [log(M/Msol)]
    """
    assert(logMstar < 13)
    return -1.79 + 0.96 * logMstar


def Sahu19(logmstar):
    """
    define the Sahu+19 bulge mass - BH mass relation
    https://ui.adsabs.harvard.edu/abs/2019ApJ...876..155S/abstract

    Parameters
    ----------
    logmstar : np.ndarray
        logarithm of bulge mass [log(M/Msol)]

    Returns
    -------
    : np.ndarray
        predicted log of bh mass [log(M/Msol)]
    """
    return 1.27 * (logmstar - np.log10(5e10)) + 8.41


def Scott13(logmstar, cored=False):
    """
    define the Scott+13 M_bh - M_bulge scaling relations for cored and non-
    cored galaxies
    https://ui.adsabs.harvard.edu/abs/2013ApJ...768...76S/abstract


    Parameters
    ----------
    logmstar : np.ndarray
        logarithm of bulge mass [log(M/Msol)]
    cored : bool, optional
        is the galaxy cored?, by default False

    Returns
    -------
    : np.ndarray
        predicted log of bh mass [log(M/Msol)]
    """
    if cored:
        return 0.97 * (logmstar - np.log10(3e11)) + 9.27
    else:
        return 2.22 * (logmstar - np.log10(3e10)) + 7.89
