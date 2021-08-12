import numpy as np

__all__ = ['Haring04', 'Magorrian98', 'Sahu19', 'Scott13']


def Haring04(logMstar):
    '''
    define Haring+04 BH-Bulge fit function

    Parameters
    ----------
    logMstar: log of bulge mass [Msol]

    Return
    ------
    predicted log of bh mass [Msol]
    '''
    assert(logMstar < 13)
    return 8.20 + 1.12 * (logMstar - 11)


def Magorrian98(logMstar):
    """
    define Magorrian+98 BH-Bulge fit function

    Parameters
    ----------
    logMstar = log of bulge mass [Msol]

    Returns
    -------
    predicted log of bh mass [Msol]
    """
    assert(logMstar < 13)
    return -1.79 + 0.96 * logMstar


def Sahu19(logmstar, u='Ks'):
    """
    define the Sahu+19 bulge mass - BH mass relation
    # TODO: define the other two u values

    Parameters
    ----------
    logmstar: log of bulge (spheroidal) mass
    u: one of Ks, scaling factor

    Returns
    -------
    log BH Mass
    """
    if u == 'Ks':
        uval = 10**(-0.06 * (logmstar-10) - 0.06)
    else:
        raise NotImplementedError('Only Ks currently implemented')
    return 1.27 * (logmstar - np.log10(uval * 5e10)) + 8.41


def Scott13(logmstar, cored=False):
    """
    define the Scott+13 M_bh - M_bulge scaling relations for cored and non-
    cored galaxies
    Parameters
    ----------
    logmstar: stellar mass, log Msol
    cored: (bool), is the galaxy cored?

    Returns
    -------
    log of BH mass
    """
    if cored:
        return 0.97 * (logmstar - np.log10(3e11)) + 9.27
    else:
        return 2.22 * (logmstar - np.log10(3e10)) + 7.89
