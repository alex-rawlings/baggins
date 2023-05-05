import numpy as np
import scipy.spatial.distance
from ..env_config import _cmlogger

_logger = _cmlogger.copy(__file__)


__all__ = ["radial_separation", "volume_sphere", "density_sphere", "angle_between_vectors"]


def radial_separation(p1, p2=0):
    """
    Determine the radial separation between two particles using scipy.spatial
    function cdist. Note that no unit conversions are performed.

    Parameters
    ----------
    p1 : np.ndarray
        particle 1 position coordinates
    p2 : np.ndarray, float, optional
        particle 2 position coordinates, or a float that specifies the same position along each axis, by default 0 (the origin)

    Returns
    -------
    : np.ndarray
        radial separation (or alternatively, magnitude) between particles
    """
    p1 = np.atleast_2d(p1)
    return scipy.spatial.distance.cdist(p1-p2, [[0]*p1.shape[-1]]).ravel()


def volume_sphere(r):
    """
    Determine the volume of a sphere of radius r.

    Parameters
    ----------
    r : np.ndarray
        radius

    Returns
    -------
    : np.ndarray
        volume, in units of r[units]^3
    """
    return 4*np.pi/3 * r**3


def density_sphere(M, r):
    """
    Determine the average density of a spherical volume (i.e., no radial 
    dependence).

    Parameters
    ----------
    M : np.ndarray
        mass
    r : np.ndarray
        radius of sphere

    Returns
    -------
    : np.ndarray
        density in units of M[units]/r[units]^3
    """
    V = volume_sphere(r)
    return M/V


def angle_between_vectors(a,b):
    """
    Determine the angle between two vectors, following the dot product. The dot product is applied along each row of an MxN array input.

    Parameters
    ----------
    a : array-like
        coordinates of vector 1
    b : array-like
        coordinates of vector 2

    Returns
    -------
    : array-like
        angle (in radians) between vectors a and b
    """
    try:
        assert a.shape == b.shape
    except AssertionError:
        _logger.logger.exception(f"Input a ({a.shape}) and b ({b.shape}) must have the same shape!", exc_info=True)
        raise
    a_hat = a / radial_separation(a)[:,np.newaxis]
    b_hat = b / radial_separation(b)[:,np.newaxis]
    return np.arccos(np.sum(a_hat*b_hat, axis=-1))

