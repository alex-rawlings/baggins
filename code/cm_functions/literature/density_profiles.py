import numpy as np
import scipy.optimize


__all__ = ['Dehnen', 'fit_Dehnen_profile', 'halfMassDehnen', 'Terzic05']


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


def Terzic05(r, rhob, rb, n, g, b, Re, a=100):
    """
    Define a density function for a cored system, with overall sersic profile
    as taken from Terzic+05
    # TODO: change mass at break radius to mass deficit?

    Parameters
    ----------
    r = radius to evaluate at [kpc]
    rhob = density at break radius
    rb = break radius [kpc]
    n = sersic index
    g = inner core density slope gamma
    b = sersic b parameter
    Re = effective radius [kpc]
    a = steepness of transition between regions

    Returns
    -------
    mass density
    """
    p = 1.0 - 0.6097/n + 0.05563/n**2
    rho_prime = rhob * 2**((p-g)/a) * (rb/Re)**p * np.exp(b*(2**(1/a) *rb/Re)**(1/n))
    Re_term = (r**a + rb**a)**(1/a) / Re
    return rho_prime * (1 + (rb/r)**a)**(g/a) * (Re_term**-p * np.exp(-b*Re_term**(1/n)))
