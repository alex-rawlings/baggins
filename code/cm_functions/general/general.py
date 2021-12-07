import numpy as np
import scipy.optimize, scipy.special, scipy.interpolate
import warnings

__all__ = ["arcsec_to_kpc", "sersic_b_param", "xval_of_quantity"]


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


def xval_of_quantity(val, xvec, yvec, xsorted=False, initial_guess=None, root_kwargs={}):
    """
    Find the value in a set of independent observations corresponding to a
    dependent observation. For example, the time corresponding to a particular
    radius value. Linear interpolation is done to create a function
    y = f(x), on which root finding is performed. 

    Parameters
    ----------
    val: y-value to determine the corresponding x-value for
    xvec: 1D array of independent observations
    yvec: 1D array of dependent observations
    xsorted (bool): are values in xvec monotonically increasing? (parsed to
                    interp1d)
    initial_guess: [a,b], where a and b specify the bounds within which val
                   should occur. Must be "either side" of val. Default (None)
                   sets [a, b] = [xvec[0], xvec[-1]].
    root_kwargs: dict of other keyword arguments to be parsed to the root
                 finding algorithm (brentq method of scipy)
    
    Returns
    -------
    xval: value of independent observations corresponding to the observed 
          dependent observation value
    

    Raises
    ------
    UserWarning: if the root-finding algorithm did not converge
    TODO should the method terminate instead?
    """
    #create the linear interpolating function
    f = scipy.interpolate.interp1d(xvec, yvec-val, assume_sorted=xsorted)
    if initial_guess is None:
        initial_guess = [xvec[0], xvec[-1]]
    xval, rootresult = scipy.optimize.brentq(f, *initial_guess, full_output=True, **root_kwargs)
    if not rootresult.converged:
        warnings.warn("The root-finding did not converge after {} iterations! The input <val> may not be in the domain specified by <xvec>.".format(rootresult.iterations))
    return xval

