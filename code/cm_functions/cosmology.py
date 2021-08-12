import numpy as np
import scipy.constants

__all__ = ['cosmology', 'time2z']


"""define the assumed cosmology constants, taken from planck 2018"""
cosmology = dict(
                    h = 0.6736,
                    omega_L = 0.6847,
                    omega_M = 0.3153
)


def time2z(t, H0=100*cosmology['h'], pres=False):
    """
    Estimate the redshift for a corresponding cosmic time
    from Carmeli 2008

    Parameters
    ----------
    t: cosmic time since big bang (Gyr) -> pres = False
    t: time before present (Gyr) -> pres = True
    H0: Hubble constant in km/s/Mpc
    pres: bool, t is given from today not Big Bang

    Returns
    -------
    z: redshift
    """
    if pres:
        #determine cosmic time
        ## TODO: make this better dependent on cosmology
        t = 13.8 - t
    #convert t [Gyr] to s
    t = t * scipy.constants.year * 1e9
    H0 = H0 / (1e3 * scipy.constants.parsec) #convert km/s/Mpc -> Hz
    z = np.sqrt(2 / (H0 * t) - 1) - 1
    return z
