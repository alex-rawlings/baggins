import numpy as np
import pygad

__all__ = ['project_orthogonal', 'spherical_components']


def project_orthogonal(vec, proj_vec=None):
    """
    Project a vector orthogonal to another vector. Projecting with either a
    single vector, and projecting each vector with a unique projection, is 
    supported.

    Parameters
    ----------
    vec: array of vectors to project
    proj_vec: a single vector or array of vectors which we project orthogonal to

    Returns
    -------
    the orthognal projection
    """
    if proj_vec is None:
        #set the default to the x-z plane
        proj_vec = np.array([1., 0., 1.])
    else:
        if len(proj_vec.shape) == 1:
            proj_vec = proj_vec[np.newaxis]
        proj_vec /= pygad.utils.dist(proj_vec)[:, np.newaxis]
    return vec - proj_vec * (np.sum(vec * proj_vec, axis=-1))[:, np.newaxis]


def spherical_components(R, v):
    """
    Convert a set of Cartesian values to spherical values.

    Parameters
    ----------
    R: array of Cartesian position coordinates
    v: array of values to convert to spherical coordinates

    Returns
    -------
    tuple (r, theta, phi) of spherical components
    """
    r = pygad.utils.dist(R) #the radial distance
    r_ = R / r[:,np.newaxis]
    theta = np.arccos(r_[:,2])
    phi = np.arctan2(r_[:,1], r_[:,0])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    theta_ = np.stack([cos_theta * cos_phi,
                       cos_theta * sin_phi, 
                       -sin_theta], axis=-1)
    phi_ = np.stack([-sin_phi, 
                     cos_phi,
                     np.zeros_like(phi)], axis=-1)
    return np.sum(r_ * v, axis=-1), np.sum(theta_ * v, axis=-1), np.sum(phi_ * v, axis=-1)