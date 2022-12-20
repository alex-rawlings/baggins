import os.path
import numpy as np
import scipy.optimize, scipy.special, scipy.interpolate
from time import time
from ..env_config import _cmlogger

__all__ = ["arcsec_to_kpc", "sersic_b_param", "xval_of_quantity", "set_seed_time", "get_idx_in_array", "get_unique_path_part"]

_logger = _cmlogger.copy(__file__)


def arcsec_to_kpc(dist_mod, angle_in_arcsec):
    """
    Convert arcseconds to kpc using a distance modulus

    Parameters
    ----------
    dist_mod : float
        distance modulus
    angle_in_arcsec : float
        angle to convert

    Returns
    -------
    : float
        distance [kpc]
    """
    return np.tan(np.radians(angle_in_arcsec/3600)) * 10**(1+dist_mod/5)/1e3


def sersic_b_param(n):
    """
    Determine the b parameter in the Sersic function
    search interval given by n=[0.5,20] -> 2n-0.33+0.009876/n

    Parameters
    ----------
    n : float
        sersic index

    Returns
    -------
    : float
        sersic b parameter
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
    val : float
        y-value to determine the corresponding x-value for
    xvec : np.ndarray
        independent observations
    yvec : np.ndarray
        dependent observations
    xsorted : bool, optional
        are values in xvec monotonically increasing? (parsed to interp1d), by 
        default False
    initial_guess : list, optional
        [a,b], where a and b specify the bounds within which val should occur. 
        Must be "either side" of val. By default None, sets [a, b] = [xvec[0], 
        xvec[-1]]
    root_kwargs : dict, optional
        other keyword arguments to be parsed to the root finding algorithm 
        (scipy.optimize.brentq), by default {}

    Returns
    -------
    xval : float
        value of independent observations corresponding to the observed 
        dependent observation value
    """
    #create the linear interpolating function
    f = scipy.interpolate.interp1d(xvec, yvec-val, assume_sorted=xsorted)
    if initial_guess is None:
        initial_guess = [xvec[0], xvec[-1]]
    xval, rootresult = scipy.optimize.brentq(f, *initial_guess, full_output=True, **root_kwargs)
    if not rootresult.converged:
        # TODO should the method terminate instead?
        _logger.logger.warning(f"The root-finding did not converge after {rootresult.iterations} iterations! The input <val> may not be in the domain specified by <xvec>.")
    return xval


def set_seed_time():
    """
    Create a random number generator seed that is the inverse of the current
    time.

    Returns
    -------
    : int
        seed
    """
    s = f"{int(time())}"[::-1]
    return int(s)


def get_idx_in_array(t, tarr):
    """
    Get the index of a value within an array. If multiple matches are found, 
    the first is returned (following np.argmin method)

    Parameters
    ----------
    t : int, float
        value to search for
    tarr : array-like
        array to search within
    Returns
    -------
    int
        index of t in tarr

    Raises
    ------
    AssertionError
        value to search for is a nan
    AssertionError
        if index is 0 or the last index of the array
    """
    try:
        assert not np.isnan(t)
    except AssertionError:
        _logger.logger.exception("t must not be nan", exc_info=True)
        raise
    try:
        idx = np.nanargmin(np.abs(tarr-t))
        if idx == len(tarr)-1:
            s = "large"
            raise AssertionError
        elif idx == 0:
            s = "smalle"
            raise AssertionError
        else:
            return idx
    except AssertionError:
        _logger.logger.exception(f"Value is {s}r than the {s}st array value!", exc_info=True)
        raise
    except ValueError:
        _logger.logger.exception(f"Array tarr has value {np.unique(tarr)}")
        raise


def get_unique_path_part(path_list):
    """
    Determine the parts of a path that are unique in a set of paths

    Parameters
    ----------
    str_list : list
        list of paths

    Returns
    -------
    unique_parts : list
        list of unique parts of the paths
    """
    try:
        assert len(path_list) > 1:
    except AssertionError:
        _logger.logger.exception(f"Path list must contain more than 1 path!", exc_info=True)
        raise
    common_path_len = len(os.path.commonpath(path_list))
    unique_parts = []
    for s in path_list:
        unique_parts.append(s[common_path_len:].lstrip("/"))
    return unique_parts

