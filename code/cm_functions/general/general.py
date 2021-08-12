import numpy as np
import scipy.optimize, scipy.special

__all__ = ['arcsec_to_kpc', 'sersic_b_param']


def arcsec_to_kpc(dist_mod, angle_in_arcsec):
    """
    Convert arcseconds to kpc using a distance modulus
    Parameters
    ----------
    dist_mod: distance modulus
    angle_in_arcsec: angle to convert

    Returns
    -------
    distance in kpc
    """
    return np.tan(np.radians(angle_in_arcsec/3600)) * 10**(1+dist_mod/5)/1e3


def sersic_b_param(n):
    """
    Determine the b parameter in the Sersic function
    search interval given by n=[0.5,20] -> 2n-0.33+0.009876/n

    Parameters
    ----------
    n: sersic index

    Returns
    -------
    b parameter
    """
    return scipy.optimize.toms748(lambda t: 2*scipy.special.gammainc(2*n, t)-1, 0.6, 19.9)
