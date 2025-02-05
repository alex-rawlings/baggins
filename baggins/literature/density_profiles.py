import copy
import numpy as np
import scipy.optimize
import pygad
from baggins.general.general import sersic_b_param


__all__ = [
    "Dehnen",
    "fit_Dehnen_profile",
    "halfMassDehnen",
    "Terzic05",
    "fit_Terzic05_profile",
    "core_Sersic_profile",
]


def Dehnen(r, a, g, M):
    """
    Calculate the Dehnen density profile

    Parameters
    ----------
    r : np.ndarray
        radial values [kpc]
    a : float
        scale radius [kpc]
    g : float
        shape profile
    M : float
        total mass

    Returns
    -------
    : np.ndarray
        3D mass density profile
    """
    return (3 - g) * M / (4 * np.pi) * a / (r**g * (r + a) ** (4 - g))


# TODO should fitting routines be incorporated into a more general method?
def fit_Dehnen_profile(radii, density, total_mass, bounds=([1, 0], [1000, 3])):
    """
    Fit a Dehnen profile to some data using the scipy.optimize library.

    Parameters
    ----------
    radii: radii values of the data
    density: 3D density values of the data (not surface density)
    total_mass: total mass in units consistent with radii and density
    bounds: parameter limits as a tuple of two lists of the form
            ([lower scale radius, lower gamma], [upper scale radius, upper
            gamma])

    Returns
    -------
    param_best: list of best-fit parameters
    """
    # fit the curve
    param_best, param_cov = scipy.optimize.curve_fit(
        lambda r, a, g: Dehnen(r, a, g, total_mass), radii, density, bounds=bounds
    )
    return param_best


def halfMassDehnen(a, g):
    """
    Determine analytical half mass radius from Dehnen sphere
    https://ui.adsabs.harvard.edu/abs/2018ApJ...864..113R/abstract

    Parameters
    ----------
    a : float
        scale radius [kpc]
    g : float
        shape parameter

    Returns
    -------
    rhm : float
        3D half mass radius [kpc]
    re : float
        (crudely) estimated effective radius ]kpc
    """
    rhm = a * (2 ** (1 / (3 - g)) - 1) ** (-1)
    re = 0.75 * rhm
    return rhm, re


def sersic_b_param_approx(n):
    """
    Approximate value of the Sersic b parameter

    Parameters
    ----------
    n : _type_
        _description_

    Returns
    -------
    float
        value of the b parameter
    """
    # assert n > 0.5 and n < 10
    return 2 * n - 0.333333333 + 0.009876 / n


def Terzic05(r, rhob, rb, n, g, Re, b=None, a=100, mode="own"):
    """
    Define a density function for a cored system, with overall sersic profile
    as taken from Terzic+05
    https://ui.adsabs.harvard.edu/abs/2005MNRAS.362..197T/abstract

    Parameters
    ----------
    r : np.ndarray
        radial values
    rhob : float
        density at break radius
    rb : float
        break radius [kpc]
    n : float
        sersic index
    g : float
        inner core density slope gamma parameter
    Re : float, pygad.UnitArr
        effective radius [kpc]
    b : float, optional
        sersic b parameter, by default None (determined internally)
    a : float, optional
        steepness of transition between regions, by default 100
    mode : str, optional
        how the function is called ("own" for general use, "fit" for fitting
        methods), by default "own"

    Returns
    -------
    : np.ndarray
        3D mass density profile
    """
    assert mode in ["own", "fit"]
    if b is None:
        if mode == "own":
            b = sersic_b_param(n)
        else:
            # use an approximation when fitting
            b = sersic_b_param_approx(n)
    if isinstance(Re, pygad.UnitArr):
        Re = Re.view(np.ndarray)
    p = 1.0 - 0.6097 / n + 0.05563 / n**2
    rho_prime = (
        rhob
        * 2 ** ((p - g) / a)
        * (rb / Re) ** p
        * np.exp(b * (2 ** (1 / a) * rb / Re) ** (1 / n))
    )
    Re_term = (r**a + rb**a) ** (1 / a) / Re
    return (
        rho_prime
        * (1 + (rb / r) ** a) ** (g / a)
        * (Re_term**-p * np.exp(-b * Re_term ** (1 / n)))
    )


def fit_Terzic05_profile(r, density, Re=None, p0=[1e2, 1, 4, 1, 1], **kwargs):
    """
    Fit a Terzic05 profile to some data using the scipy.optimize library.

    Parameters
    ----------
    r : array-like
        radii to fit
    density : array-like
        3D density of galaxy as a function of radius
    Re : flaot, optional
        effective radius of galaxy, by default None
    p0 : list, optional
        initial parameter guesses, by default [1e2, 1, 4, 1, 1]

    Returns
    -------
    param_best: dict of best-fit parameters
    """
    p0 = copy.copy(p0)
    if Re is None:
        p0.append(7)
        bounds = ((0, 0, 0.2, 0, 0, 0), (np.inf, 10, 20, np.inf, 15, 20))
        f = lambda r, rhob, rb, n, g, a, Re: Terzic05(
            r, rhob, rb, n, g, Re, a, mode="fit"
        )
    else:
        bounds = ((0, 0, 0, 0, 0), (np.inf, 5, 20, np.inf, 15))
        f = lambda r, rhob, rb, n, g, a: Terzic05(r, rhob, rb, n, g, Re, a, mode="fit")
    popt, param_cov = scipy.optimize.curve_fit(
        f, r, density, bounds=bounds, p0=p0, **kwargs
    )
    param_best = {
        "rhob": popt[0],
        "rb": popt[1],
        "n": popt[2],
        "g": popt[3],
        "a": popt[4],
    }
    if Re is None:
        param_best["Re"] = popt[5]
    return param_best


def core_Sersic_profile(r, Re, rb, Ib, n, gamma, alpha=10.0):
    """
    Core Sersic profile from Graham 2003. Note this is a projected (2D) profile!
    Note that no unit-consistency checks are done.

    Parameters
    ----------
    r : array-like
        radial values to determine profile for
    Re : float
        effective radius
    rb : float
        core radius
    Ib : float
        normalising intensity
    n : float
        sersic index
    gamma : float
        inner core slope index
    alpha : float, optional
        transition index, by default 10

    Returns
    -------
    : array-like
        projected density profile
    """
    bn = sersic_b_param_approx(n)
    Ib_ = Ib * 2 ** (-gamma / alpha) * np.exp(bn * (2 ** (1 / alpha) * rb / Re))
    return (
        Ib_
        * (1 + (rb / r) ** alpha) ** (gamma / alpha)
        * np.exp(-bn * ((r**alpha + rb**alpha) / Re**alpha) ** (1 / (alpha * n)))
    )
