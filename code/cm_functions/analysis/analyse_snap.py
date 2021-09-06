import numpy as np
import scipy.linalg
import pygad

__all__ = ['get_com_of_each_galaxy', 'get_com_velocity_of_each_galaxy', 'get_galaxy_axis_ratios']


def get_com_of_each_galaxy(snap, initial_radius=10, masks=None, verbose=True, min_particle_count=10):
    """
    Determine the centre of mass of each galaxy in the simulation, assuming each
    galaxy has a single SMBH near its centre.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    initial_radius: initial radius for shrinking sphere method
    masks: pygad masks to apply to the (sub) snapshot
    verbose: print verbose output
    min_particle_count: stop the shrinking_sphere method when this many
                        particles or less are contained

    Returns
    -------
    coms: dict with n keys, where each key corresponds to the centre of mass
          of each galaxy
    """
    assert(snap._phys_units_requested)
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    #prepare dict that will hold the centre of masses
    coms = dict()
    for ind, idx in enumerate(snap.bh['ID']):
        if verbose:
            print('Finding CoM by centring on BH {}'.format(ind))
        if masks is not None:
            subsnap = snap.stars[masks[idx]]
        else:
            subsnap = snap.stars
        coms[idx] = pygad.analysis.shrinking_sphere(subsnap, snap.bh['pos'][snap.bh['ID']==idx, :], initial_radius, stop_N=min_particle_count)
    return coms


def get_com_velocity_of_each_galaxy(snap, xcom, masks=None, min_particle_count=5e4, verbose=True):
    """
    Determine the centre of mass velocity of each galaxy in the simulation,
    assuming each galaxy has an SMBH near its centre.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    xcom: dict of CoM coordinates for each galaxy, assumes the dict keys are the
          BH particle IDs
    masks: pygad masks to apply to the (sub) snapshot
    min_particle_count: minimum number of particles to be contained in the
                        sphere
    verbose: print verbose output

    Returns
    -------
    vcoms: dict with n keys, where each key corresponds to the centre of mass
          velocity of each galaxy
    """
    assert(snap._phys_units_requested)
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    #prepare the dict that will hold the velocity centre of masses
    vcoms = dict()
    for ind, idx in enumerate(snap.bh['ID']):
        if masks is not None:
            subsnap = snap.stars[masks[idx]]
        else:
            subsnap = snap.stars
        #make a ball about the CoM
        ball_radius = np.sort(pygad.utils.dist(subsnap['pos'], xcom[idx]))[int(min_particle_count)]
        if verbose:
            print('Maximum radius for velocity CoM set to {} kpc'.format(ball_radius))
        ball_mask = pygad.BallMask(pygad.UnitQty(ball_radius, 'kpc'), center=xcom[idx])
        vcoms[idx] = pygad.analysis.mass_weighted_mean(subsnap[ball_mask], qty='vel')
    return vcoms


def get_galaxy_axis_ratios(snap, xcom=None, vcom=None, family='stars', return_eigenvectors=False):
    """
    Determine the axis ratios b/a and c/a of a galaxy
    """
    if xcom is None:
        xcom = pygad.UnitArr([0,0,0], 'kpc')
    elif not isinstance(xcom, pygad.UnitArr):
        xcom = pygad.UnitArr(xcom, 'kpc')
    if vcom is None:
        vcom = pygad.UnitArr([0,0,0], 'km/s')
    elif not isinstance(vcom, pygad.UnitArr):
        vcom = pygad.UnitArr(vcom, 'km/s')
    
    subsnap = getattr(snap, family)
    #move to CoM coordinates
    subsnap['pos'] -= xcom
    subsnap['vel'] -= vcom
    #get the reduced inertia tensor
    rit = pygad.analysis.reduced_inertia_tensor(subsnap)
    eigen_vals, eigen_vecs = scipy.linalg.eig(rit)
    #need to sort the eigenvalues
    sorted_idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_idx]
    eigen_vecs = eigen_vecs[:, sorted_idx]
    axis_ratios = np.sqrt(eigen_vals[1:]/eigen_vals[0])
    if return_eigenvectors:
        return axis_ratios, eigen_vecs
    else:
        return axis_ratios
