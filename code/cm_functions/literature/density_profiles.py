import numpy as np
import scipy.optimize
from ..general import sersic_b_param


__all__ = ['Dehnen', 'fit_Dehnen_profile', 'halfMassDehnen', 'Terzic05', "fit_Terzic05_profile"]


def Dehnen(r, a, g, M):
    '''
    Calculate the Dehnen density profile

    Parameters
    ----------
    r: radius [kpc]
    a: scale radius [kpc]
    g: shape profile
    M: total Mass [Msol]

    Returns
    -------
    rho: density
    '''
    rho = (3 - g) * M / (4 * np.pi) * a / (r**g * (r + a)**(4 - g))
    return rho


def fit_Dehnen_profile(radii, density, total_mass, bounds=([1, 0], [1000,3])):
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
    #fit the curve
    param_best, param_cov = scipy.optimize.curve_fit(lambda r, a, g: Dehnen(r, a, g, total_mass), radii, density, bounds=bounds)
    return param_best


def halfMassDehnen(a, g):
    """
    Determine analytical half mass radius from Dehnen sphere
    taken from Rantala+18

    Parameters
    ----------
    a: scale radius (generally in kpc)
    g: Dehnen gamma

    Returns
    -------
    rhm: half mass radius [kpc]
    re: effective radius approximate [kpc]
    """
    rhm = a * (2**(1/(3-g)) - 1)**(-1)
    re = 0.75 * rhm
    return rhm, re


def Terzic05(r, rhob, rb, n, g, Re, b=None, a=100, mode="own"):
    """
    Define a density function for a cored system, with overall sersic profile
    as taken from Terzic+05
    # TODO: change mass at break radius to mass deficit?

    Parameters
    ----------
    r: radius to evaluate at [kpc]
    rhob: density at break radius
    rb: break radius [kpc]
    n: sersic index
    g: inner core density slope gamma
    Re: effective radius [kpc]
    b: sersic b parameter (set to None to determine within this method)
    a: steepness of transition between regions
    mode: how the function is called ('own' for general use, 'fit for fitting
          methods)

    Returns
    -------
    mass density
    """
    assert mode in ["own", "fit"]
    if b is None:
        if mode == "own":
            b = sersic_b_param(n)
        else:
            #use an approximation when fitting
            b = 2*n - 0.33 + 0.009876/n

    p = 1.0 - 0.6097/n + 0.05563/n**2
    rho_prime = rhob * 2**((p-g)/a) * (rb/Re)**p * np.exp(b*(2**(1/a) *rb/Re)**(1/n))
    Re_term = (r**a + rb**a)**(1/a) / Re
    return rho_prime * (1 + (rb/r)**a)**(g/a) * (Re_term**-p * np.exp(-b*Re_term**(1/n)))


def fit_Terzic05_profile(r, density, Re, p0=[1e2, 1, 4, 1, 1], **kwargs):
    """
    Fit a Terzic05 profile to some data using the scipy.optimize library.

    Parameters
    ----------
    r: array of radii to fit
    density: 3D density of galaxy as a function of radius
    Re: effective radius of galaxy
    p0: list of initial parameter guesses

    Returns
    -------
    param_best: dict of best-fit parameters
    """
    bounds = ((0, 0, 0, 0, 0), (np.inf, 5, 20, np.inf, 15))
    f = lambda r, rhob, rb, n, g, a: Terzic05(r, rhob, rb, n, g, Re, a, mode="fit")
    popt, param_cov = scipy.optimize.curve_fit(f, r, density, bounds=bounds, p0=p0, **kwargs)
    param_best = {"rhob": popt[0], "rb": popt[1], "n":popt[2], "g": popt[3], "a":popt[4]}
    return param_best