import numpy as np
import scipy.constants, scipy.optimize
from .cosmology import *

__all__ = ["time2z"]


def time2z(t, H0=100 * cosmology["h"], pres=False):
    """
    Estimate the redshift for a corresponding cosmic time
    from Carmeli 2008

    Parameters
    ----------
    t : float
        cosmic time since big bang [Gyr] -> pres = False
        time before present [Gyr] -> pres = True
    H0 : float, optional
        Hubble constant in km/s/Mpc, by default 100*cosmology['h']
    pres : bool, optional
        t is given from today not Big Bang?, by default False

    Returns
    -------
    : float
        redshift
    """
    if pres:
        # determine cosmic time
        # TODO: make this better dependent on cosmology
        t = 13.8 - t
    # convert t [Gyr] to s
    t = t * scipy.constants.year * 1e9
    H0 = H0 / (1e3 * scipy.constants.parsec)  # convert km/s/Mpc -> Hz
    return np.sqrt(2 / (H0 * t) - 1) - 1


# TODO: conversions for luminosity distance -> redshift
def convert_lum_dist_2_z(lum_dist, cosmology=cosmology):
    """
    Convert luminosity distance to redshift using the bisection method

    Parameters
    ----------
    lum_dist: luminosity distance in kpc
    cosmology: dict of cosmology parameters

    Returns
    -------
    estimated redshift
    """
    try:
        # assume in most cases we'll be dealing with modest redshifts
        # so set an aggressive upper limit
        return scipy.optimize.bisect(
            lambda z: lum_dist - luminosity_distance(z, cosmology), 0, 10
        )
    except ValueError:
        # we are dealing with a higher redshift
        return scipy.optimize.bisect(
            lambda z: lum_dist - luminosity_distance(z, cosmology), 0, cosmology["zeq"]
        )
