import numpy as np
import scipy.spatial.distance
from scipy.spatial.transform import Rotation
from ..env_config import _cmlogger

_logger = _cmlogger.getChild(__name__)


__all__ = [
    "radial_separation",
    "volume_sphere",
    "density_sphere",
    "angle_between_vectors",
    "rotate_vec1_to_vec2",
]


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
    return scipy.spatial.distance.cdist(p1 - p2, [[0] * p1.shape[-1]]).ravel()


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
    return 4 * np.pi / 3 * r**3


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
    return M / V


def angle_between_vectors(a, b):
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
        _logger.exception(
            f"Input a ({a.shape}) and b ({b.shape}) must have the same shape!",
            exc_info=True,
        )
        raise
    a_hat = a / radial_separation(a)[:, np.newaxis]
    b_hat = b / radial_separation(b)[:, np.newaxis]
    return np.arccos(np.sum(a_hat * b_hat, axis=-1))


def rotate_vec1_to_vec2(v1, v2, tol=1e-9):
    """
    Create a rotation transform from vector1 to vector2 using quaternions.
    The input vectors can either be a single vector or an array of vectors,
    in which case the rotation for each vector in the array v1 to the
    corresponding vector in v2 is determined (see scipy docs for specifics).
    This function is largely based on the function
    `rotate_to_from()`
    in the merger_ic_generator package.

    Parameters
    ----------
    v1 : array-like
        vectors to rotate
    v2 : array-like
        vectors to rotate to
    tol : float, optional
        zero-comparison tolerance, by default 1e-9

    Returns
    -------
    scipy.spatial.transform.Rotation
        quaternion rotation to rotate v1 to v2
    """
    # normalise vectors
    v1 = v1 / np.linalg.norm(v1, axis=0)
    v2 = v2 / np.linalg.norm(v2, axis=0)

    # check if vectors are parallel or antiparallel
    dot_product = np.sum(v1 * v2, axis=-1)
    parallel_mask = np.abs(dot_product - 1) < tol
    antiparallel_mask = np.abs(dot_product + 1) < tol
    ok_mask = np.logical_not(parallel_mask + antiparallel_mask, dtype=bool)
    # initialise arrays
    sin_th2 = np.full((len(v1), 1), np.nan)
    cos_th2 = np.full_like(sin_th2, np.nan)
    arg = np.full((len(v1), 4), np.nan)

    def _construct_axes(a, b):
        """helper function to construct the rotation axis"""
        cross = np.cross(b, a, axis=-1)
        cross /= np.linalg.norm(cross, axis=-1, keepdims=True)
        cross = np.atleast_2d(cross)
        return cross

    # get the rotation axis for 'regular' pairs
    v2_ok = v2 if v2.ndim == 1 else v2[ok_mask]
    cross = _construct_axes(v2_ok, v1[ok_mask])
    # get the rotation axis for antiparallel pairs
    # we can rotate about any axis, so use the z-axis as a helper to construct
    # the rotation basis from
    cross_antiparallel = _construct_axes([0, 0, 1], v1[antiparallel_mask])
    # get the angle to rotate through
    half_theta = 0.5 * np.arccos(np.sum(v2_ok * v1[ok_mask], axis=-1))
    sin_th2[ok_mask] = np.sin(half_theta)[:, np.newaxis]
    cos_th2[ok_mask] = np.cos(half_theta)[:, np.newaxis]
    arg[ok_mask, :-1] = sin_th2[ok_mask] * cross
    arg[:, -1] = cos_th2.flatten()
    arg[parallel_mask, :] = [0, 0, 0, 1]
    arg[antiparallel_mask, :] = np.append(cross_antiparallel, 0)

    # construct the quaternion
    q = Rotation.from_quat(arg)
    return q
