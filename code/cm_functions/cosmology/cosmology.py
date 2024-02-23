import numpy as np
import scipy.constants
import scipy.integrate

__all__ = ["cosmology", "angular_diameter_distance", "get_a0r", "luminosity_distance"]


"""
Define the assumed cosmology constants, taken from planck 2018

Attributes
----------
h: value of Hubble constant H0/100
omega_L: cosmic density parameter for dark energy
omega_M: cosmic density parameter for non-relativistic matter
zeq: redshift of radiation-matter equality
"""
cosmology = dict(h=0.6736, omega_L=0.6847, omega_M=0.3153, zeq=3402)


def angular_diameter_distance(z, cosmology=cosmology):
    """
    Determine the angular diameter distance for a flat universe

    Parameters
    ----------
    z : float
        redshift
    cosmology : dict, optional
        cosmological parameters, by default cosmology

    Returns
    -------
    : float
        angular diameter distance [kpc]
    """
    return get_a0r(z, cosmology) / (1 + z)


def get_a0r(z, cosmology=cosmology):
    """
    Determine the value a_0*r from MBW eq. 3.106, for use in cosmological
    distance calculations

    Parameters
    ----------
    z : float
        redshift
    cosmology : dict, optional
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
    assert z < cosmology["zeq"]
    Ez = lambda z1: np.sqrt(cosmology["omega_L"] + cosmology["omega_M"] * (1 + z1) ** 3)
    return scipy.integrate.quad(Ez, 0, z) * scipy.constants.c / (100 * cosmology["h"])


def luminosity_distance(z, cosmology=cosmology):
    """
    Determine the luminosity distance for a flat universe

    Parameters
    ----------
    z : float
        redshift
    cosmology : dict, optional
        cosmological parameters, by default cosmology

    Returns
    -------
    : float
        luminosity distance [kpc]
    """
    return get_a0r(z, cosmology) * (1 + z)
