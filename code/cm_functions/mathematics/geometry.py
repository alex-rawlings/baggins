import numpy as np
import scipy.spatial.distance

__all__ = ["radial_separation", "volume_sphere", "density_sphere"]


def radial_separation(p1, p2=0):
    """
    Determine the radial separation between two particles using scipy.spatial
    function cdist. Note that no unit conversions are performed.

    Parameters
    ----------
    p1: array of particle 1 position coordinates
    p2: array of particle 2 position coordinates

    Returns
    -------
    radial separation between particles
    """
    p1 = np.atleast_2d(p1)
    return scipy.spatial.distance.cdist(p1-p2, [[0]*p1.shape[-1]]).ravel()


def volume_sphere(r):
    """
    Determine the volume of a sphere of radius r.

    Parameters
    ----------
    r: radius

    Returns
    -------
    volume, in units of r[units]^3
    """
    return 4*np.pi/3 * r**3


def density_sphere(M, r):
    """
    Determine the average density of a spherical volume (i.e., no radial 
    dependence).

    Parameters
    ----------
    M: mass
    r: radius of sphere

    Returns
    -------
    density in units of M[units]/r[units]^3
    """
    V = volume_sphere(r)
    return M/V