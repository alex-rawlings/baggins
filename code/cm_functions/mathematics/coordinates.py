import numpy as np
from .geometry import radial_separation

__all__ = ["project_orthogonal", "set_spherical_basis", "spherical_components", "radial_separation", "cartesian_components", "convert_cartesian_to_spherical", "convert_spherical_to_cartesian"]


def project_orthogonal(vec, proj_vec=None):
    """
    Project a vector orthogonal to another vector. Projecting with either a
    single vector, and projecting each vector with a unique projection, is 
    supported.

    Parameters
    ----------
    vec : np.ndarray
        vectors to project
    proj_vec : np.ndarray, optional
        single vector or array of vectors which we project orthogonal to, by default None (projects to x-z plane)

    Returns
    -------
    : np.ndarray
        orthognal projection
    """
    if proj_vec is None:
        #set the default to the x-z plane
        proj_vec = np.array([1., 0., 1.])
    else:
        if len(proj_vec.shape) == 1:
            proj_vec = proj_vec[np.newaxis]
        proj_vec /= radial_separation(proj_vec)[:, np.newaxis]
    return vec - proj_vec * (np.sum(vec * proj_vec, axis=-1))[:, np.newaxis]


def set_spherical_basis(R):
    """
    Set the spherical coordinate basis.

    Parameters
    ----------
    R : np.ndarray
        array to use to set the spherical coordinate basis, yypically will be
        particle position vector

    Returns
    -------
    r_: np.ndarray
        radial component. This will be aligned with the Cartesian direction of
        input R
    theta_: np.ndarray
        angular inclination components orthogonal to r_. Definition as per the 
        "physicist's" (ISO) definition
    phi_: np.ndarray
        angular azimuth components orthogonal to r_. Definition as per the 
        "physicist's" (ISO) definition
    """
    r = radial_separation(R) # radial distance
    r_ = R / r[:,np.newaxis] #determine basis vectors
    theta = np.arccos(r_[:,2]) #arccos(z/r)
    phi = np.arctan2(r_[:,1], r_[:,0]) #arctan(y/x)
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
    return r_, theta_, phi_


def spherical_components(R, v):
    """
    Convert a set of Cartesian values to spherical values.

    Parameters
    ----------
    R : np.ndarray
        Cartesian position coordinates to set spherical basis
    v : np.ndarray
        values to convert to spherical coordinates

    Returns
    -------
    : (n,3) np.ndarray 
        spherical components, with columns corresponding to radius, theta, and 
        phi
    """
    r_, theta_, phi_ = set_spherical_basis(R)
    return np.stack((np.sum(r_ * v, axis=-1), np.sum(theta_ * v, axis=-1), np.sum(phi_ * v, axis=-1)), axis=-1)


def cartesian_components(R, v):
    """
    Convert a set of spherical values in (r,t,p) form to Cartesian values.
    Requires that the basis vectors used to convert to spherical coordinates 
    is known if we want to do a back-transformation.

    Parameters
    ----------
    R : np.ndarray
        Cartesian position coordinates to set spherical basis
    v : np.ndarray
        spherical values to convert to Cartesian coordinates

    Returns
    -------
    : (n,3) np.ndarray
        Cartesian components, with columns corresponding to x, y, and z
    """
    r_, theta_, phi_ = set_spherical_basis(R)
    xyz = [r_[:,i]*v[:,0] + theta_[:,i]*v[:,1] + phi_[:,i]*v[:,2] for i in range(3)]
    return np.stack((xyz[0], xyz[1], xyz[2]), axis=-1)


def convert_cartesian_to_spherical(R):
    """
    The simple transform from (x,y,z) -> (r,theta,phi). Note this does not work
    with basis vectors, so e.g. adding theta+eps will not change the magnitude
    of the vector

    Parameters
    ----------
    R : np.ndarray
        Cartesian values

    Returns
    -------
    S : (n,3) np.ndarray
        spherical values
    """
    R = np.atleast_2d(R)
    S = np.full_like(R, np.nan)
    S[:,0] = radial_separation(R)
    S[:,1] = np.arccos(R[:,2]/S[:,0])
    S[:,2] = np.arctan2(R[:,1], R[:,0])
    return S


def convert_spherical_to_cartesian(S):
    """
    The simple transform from (r,theta,phi) -> (x,y,z). Note this does not work
    with basis vectors, so e.g. adding theta+eps will not change the magnitude
    of the vector

    Parameters
    ----------
    S : np.ndarray
        spherical values

    Returns
    -------
    R : np.ndarray
        Cartesian values
    """
    S = np.atleast_2d(S)
    R = np.full_like(S, np.nan)
    R[:,0] = S[:,0]*np.sin(S[:,1])*np.cos(S[:,2])
    R[:,1] = S[:,0]*np.sin(S[:,1])*np.sin(S[:,2])
    R[:,2] = S[:,0]*np.cos(S[:,1])
    return R
