import numpy as np
import scipy.constants
import scipy.integrate

__all__ = [
    "cosmology_pars",
    "angular_diameter_distance",
    "luminosity_distance",
    "distance_modulus",
    "angular_scale"
]


"""
Define the assumed cosmology constants, taken from planck 2018

Attributes
----------
h: value of Hubble constant H0/100
omega_L: cosmic density parameter for dark energy
omega_M: cosmic density parameter for non-relativistic matter
zeq: redshift of radiation-matter equality
"""
cosmology_pars = dict(h=0.6736, omega_M=0.3153, zeq=3402)
cosmology_pars["omega_L"] = 1 - cosmology_pars["omega_M"]


def get_a0r(z, cosmo_p=cosmology_pars):
    """
    Determine the value a_0*r from MBW eq. 3.106, for use in cosmological
    distance calculations. a0 is the scale factor at the present day, which
    we take to be 1 (MBW, pg. 116).

    Parameters
    ----------
    z : float
        redshift
    cosmo_p : dict, optional
        cosmological parameters, by default cosmology

    Returns
    -------
    : float
        a0*r [kpc]

    Raises
    ------
    AssertionError
        redshift larger than redshift of matter radiation equality
    """
    assert z < cosmo_p["zeq"]
    inv_Ez = lambda z1: 1 / np.sqrt(cosmo_p["omega_L"] + cosmo_p["omega_M"] * (1 + z1) ** 3)
    # want the answer in kpc
    # c [m/s] / (km/s / Mpc) = c/1e3 [km/s] / (km/s / (kpc*1e3)) = kpc
    return scipy.integrate.quad(inv_Ez, 0, z)[0] * scipy.constants.c / (100 * cosmo_p["h"])


def angular_diameter_distance(z, cosmo_p=cosmology_pars):
    """
    Determine the angular diameter distance for a flat universe

    Parameters
    ----------
    z : float
        redshift
    cosmo_p : dict, optional
        cosmological parameters, by default cosmology_pars

    Returns
    -------
    : float
        angular diameter distance [kpc]
    """
    return get_a0r(z, cosmo_p) / (1 + z)


def luminosity_distance(z, cosmo_p=cosmology_pars):
    """
    Determine the luminosity distance for a flat universe

    Parameters
    ----------
    z : float
        redshift
    cosmo_p : dict, optional
        cosmological parameters, by default cosmology_pars

    Returns
    -------
    : float
        luminosity distance [kpc]
    """
    return get_a0r(z, cosmo_p) * (1 + z)


def distance_modulus(z, cosmo_p=cosmology_pars):
    """
    Determine the distance modulus, Eq 3.6 of Carroll & Ostlie 2017

    Parameters
    ----------
    z : float
        redshift
    cosmo_p : dict, optional
        cosmological parameters, by default cosmology_pars

    Returns
    -------
    : float
        distance modulus [mag]
    """
    # divide by 1e-2 as this is 10pc in kpc
    return 5 * np.log10(luminosity_distance(z=z, cosmo_p=cosmo_p) / 1e-2)


def angular_scale(z, cosmo_p=cosmology_pars):
    """
    Determine angular scale

    Parameters
    ----------
    z : float
        redshift
    cosmo_p : dict, optional
        cosmological parameters, by default cosmology_pars

    Returns
    -------
    : float
        angular scale [phys. kpc / arcsec]
    """
    return angular_diameter_distance(z=z, cosmo_p=cosmo_p) * np.pi / (180 * 3600)
