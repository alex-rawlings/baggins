from astropy import cosmology, units

__all__ = [
    "Hubble_parameter",
    "comoving_radial_distance",
    "cosmology_pars",
    "angular_diameter_distance",
    "luminosity_distance",
    "distance_modulus",
    "angular_scale",
]


cosmo = cosmology.Planck18

"""
Define the assumed cosmology constants, taken from planck 2018

Attributes
----------
h: value of Hubble constant H0/100
omega_L: cosmic density parameter for dark energy
omega_M: cosmic density parameter for non-relativistic matter
zeq: redshift of radiation-matter equality
"""
cosmology_pars = dict(h=cosmo.h, omega_M=cosmo.Om0, zeq=3402)
cosmology_pars["omega_L"] = 1 - cosmology_pars["omega_M"]


def _output_unit_formatter(q, u):
    """
    Helper to convert correct output units

    Parameters
    ----------
    q : astropy.units.Quantity
        quantity to convert
    u : str
        unit to conver to

    Returns
    -------
    q : astropy.units.Quantity
        quantity
    """
    if u is None:
        return q
    else:
        return q.to(units.Unit(u))


def Hubble_parameter(z, unit="1/Gyr"):
    """
    Hubble parameter as a function of redshift

    Parameters
    ----------
    z : float
        redshift to evaluate at
    unit : str, optional
        unit to convert to, by default "1/Gyr

    Returns
    -------
    : astropy.unit.Quantity
        Hubble parameter
    """
    H = cosmo.H(z=z)
    return _output_unit_formatter(H, unit)


def comoving_radial_distance(z, unit="kpc"):
    """
    Determine the value a_0*r from MBW eq. 3.106, for use in cosmological
    distance calculations. a0 is the scale factor at the present day, which
    we take to be 1 (MBW, pg. 116).

    Parameters
    ----------
    z : float
        redshift
    unit : str, optional
        unit to convert to, by default "kpc"

    Returns
    -------
    : astropy.unit.Quantity
        comoving radial distance
    """
    d = cosmo.comoving_distance(z=z)
    return _output_unit_formatter(d, unit)


def angular_diameter_distance(z, unit="kpc"):
    """
    Determine the angular diameter distance for a flat universe

    Parameters
    ----------
    z : float
        redshift
    unit : str, optional
        unit to convert to, by default "kpc"

    Returns
    -------
    : astropy.unit.Quantity
        angular diameter distance
    """
    d = cosmo.angular_diameter_distance(z=z)
    return _output_unit_formatter(d, unit)


def luminosity_distance(z, unit="kpc"):
    """
    Determine the luminosity distance for a flat universe

    Parameters
    ----------
    z : float
        redshift
    unit : str, optional
        unit to convert to, by default "kpc"

    Returns
    -------
    : astropy.unit.Quantity
        luminosity distance
    """
    d = cosmo.luminosity_distance(z=z)
    return _output_unit_formatter(d, unit)


def distance_modulus(z, unit=None):
    """
    Determine the distance modulus, Eq 3.6 of Carroll & Ostlie 2017

    Parameters
    ----------
    z : float
        redshift
    unit : str, optional
        unit to convert to, by default None

    Returns
    -------
    : astropy.unit.Quantity
        distance modulus
    """
    d = cosmo.distmod(z=z)
    return _output_unit_formatter(d, unit)


def angular_scale(z, unit="kpc/arcsec"):
    """
    Determine angular scale

    Parameters
    ----------
    z : float
        redshift
    unit : str, optional
        unit to convert to, by default "kpc/arcsec"

    Returns
    -------
    : astropy.unit.Quantity
        physical angular scale
    """
    s = 1 / cosmo.arcsec_per_kpc_proper(z=z)
    return _output_unit_formatter(s, unit)
