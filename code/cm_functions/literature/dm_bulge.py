import numpy as np
import scipy.optimize

__all__ = ["Behroozi19", "Girelli20", "Moster10"]



###### HELPER FUNCTIONS #######
def _moster10(h, M1, mM0, b, g):
    """the form of the moster 10 function for dm-stellar mass relation"""
    return 2 * mM0 * ((h / M1)**(-b) + (h / M1)**g)**(-1) * h

def _moster_helper(h, s, M1, mM0, b, g):
    """helper function for bisection method (could be an anonymous lambda
    function, but is called a few times)"""
    return s - _moster10(h, M1, mM0, b, g)


###### CALLABLE FUNCTIONS #######

def Behroozi19(sm, hm=[1e10, 1e15], z=0, plotting=False, numPoints=1000):
    """
    define the Behroozi+19 relation for stellar-halo mass
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B/abstract

    Parameters
    ----------
    sm : float
        stellar mass [Msol]
    hm : list, optional
        estimated bounds of halo mass, by default [1e10, 1e15]
    z : float, optional
        redshift, for redshift dependent scaling relation, by default 0
    plotting : bool, optional
        recover the halo mass (False), or return an array of values for 
        plotting (True)?, by default False
    numPoints : int, optional
        number of points used in bisection method (and plotting) to determine 
        halo mass from stellar mass, by default 1000

    Returns
    -------
    : float
        logarithm of halo mass [log(M/Msol)]
    : np.ndarray, optional
        logarithm of corresponding stellar masses [log(M/Msol)] if plotting is 
        True
    
    Raises
    ------
    AssertionError
        invalid input for hm
    """
    assert(len(hm)==2 and hm[0]<hm[1])
    M0 = 12.069
    Ma = 2.646
    Mlna = 2.710
    Mz = -0.431
    eps0 = -1.480
    epsa = -0.831
    epslna = -1.351
    epsz = 0.321
    alpha0 = 1.899
    alphaa = -2.901
    alphalna = -2.413
    alphaz = 0.332
    beta0 = 0.502
    betaa = -0.315
    betaz = -0.218
    delta0 = 0.397
    gamma0 = -0.867
    gammaa = -1.146
    gammaz = -0.294
    a = 1/(1+z)

    sm = np.log10(sm)
    logM1 = M0 + Ma * (a-1) - Mlna * np.log(a) + Mz * z
    epsilon = eps0 + epsa*(a-1) - epslna * np.log(a) + epsz*z
    alpha = alpha0 + alphaa * (a-1) - alphalna*np.log(a) + alphaz*z
    beta = beta0 + betaa*(a-1) + betaz*z
    logGamma = gamma0 + gammaa*(a-1) + gammaz*z
    h_mass = np.logspace(np.log10(hm[0]), np.log10(hm[1]), numPoints)
    def _behroozi19(h):
        x = np.log10(h) - logM1
        return epsilon - np.log10(10**(-alpha*x) + 10**(-beta*x)) + 10**logGamma * np.exp(-0.5 * (x/delta0)**2) + logM1
    if plotting:
        logMstar = _behroozi19(h_mass)
        return np.log10(h_mass), logMstar
    else:
        h_mass = scipy.optimize.bisect(lambda h: sm - _behroozi19(h), hm[0], hm[1], xtol=1)
        return np.log10(h_mass)


def Girelli20(sm, hm=[1e10, 1e15], z=0, plotting=False, numPoints=1000):
    """
    Define the halo-stellar mass relation from Girelli+20. Uses the same form
    as the Moster+10 relation, so the helper function _moster10 is called.
    https://ui.adsabs.harvard.edu/abs/2020A%26A...634A.135G/abstract

    Parameters
    ----------
    sm : float
        stellar mass [Msol]
    hm : list, optional
        estimated bounds of halo mass, by default [1e10, 1e15]
    z : float, optional
        redshift, for redshift dependent scaling relation, by default 0
    plotting : bool, optional
        recover the halo mass (False), or return an array of values for 
        plotting (True)?, by default False
    numPoints : int, optional
        number of points used in bisection method (and plotting) to determine 
        halo mass from stellar mass, by default 1000

    Returns
    -------
    : float
        logarithm of halo mass [log(M/Msol)]
    : np.ndarray, optional
        logarithm of corresponding stellar masses [log(M/Msol)] if plotting is 
        True
    """
    assert(len(hm)==2 and hm[0]<hm[1])
    M1 = 10**(11.83 + z * 0.18)
    mM0 = 0.047 * (z+1)**-0.40
    b = 0.92 + 0.052 * z
    g = 0.728 * (z + 1)**-0.16
    if plotting:
        #plot the relation
        h_mass = np.logspace(np.log10(hm[0]), np.log10(hm[1]), numPoints)
        return np.log10(h_mass), np.log10(_moster10(h_mass, M1, mM0, b, g))
    else:
        #determine the halo mass from the stellar mass
        h_mass = scipy.optimize.bisect(_moster_helper, hm[0], hm[1], args=(sm, M1, mM0, b, g), xtol=1)
        return np.log10(h_mass)


def Moster10(sm, hm=[1e10, 1e15], z=0, plotting=False, numPoints=1000):
    """
    Define the halo-stellar mass relation from Girelli+20. Uses the same form
    as the Moster+10 relation, so the helper function _moster10 is called.
    https://ui.adsabs.harvard.edu/abs/2010ApJ...710..903M/abstract

    Parameters
    ----------
    sm : float
        stellar mass [Msol]
    hm : list, optional
        estimated bounds of halo mass, by default [1e10, 1e15]
    z : float, optional
        redshift, for redshift dependent scaling relation, by default 0
    plotting : bool, optional
        recover the halo mass (False), or return an array of values for 
        plotting (True)?, by default False
    numPoints : int, optional
        number of points used in bisection method (and plotting) to determine 
        halo mass from stellar mass, by default 1000

    Returns
    -------
    : float
        logarithm of halo mass [log(M/Msol)]
    : np.ndarray, optional
        logarithm of corresponding stellar masses [log(M/Msol)] if plotting is 
        True
    """
    assert(len(hm)==2 and hm[0]<hm[1])
    M1 = 10**(11.88 * (z+1)**0.019)
    mM0 = 0.0282 * (z+1)**-0.72
    b = 1.06 + 0.17 * z
    g = 0.556 * (z + 1)**-0.26


    if plotting:
        #plot the relation
        h_mass = np.logspace(np.log10(hm[0]), np.log10(hm[1]), numPoints)
        return np.log10(h_mass), np.log10(_moster10(h_mass, M1, mM0, b, g))
    else:
        #determine the halo mass from the stellar mass
        h_mass = scipy.optimize.bisect(_moster_helper, hm[0], hm[1], args=(sm, M1, mM0, b, g), xtol=1)
        return np.log10(h_mass)
