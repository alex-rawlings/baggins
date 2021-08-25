import numpy as np
import scipy.linalg
import pygad

__all__ = ['get_coms_of_each_galaxy', 'get_com_velocity_of_each_galaxy', 'get_all_id_masks', 'get_id_mask']


def get_coms_of_each_galaxy(snap, initial_radius=10, masks=None, verbose=True, min_particle_count=10):
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


def get_com_velocity_of_each_galaxy(snap, xcom, masks=None, min_particle_count=1e4, verbose=True):
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


def get_all_id_masks(snap, radius=20, family='stars'):
    """
    Return a list of masks that mask the chosen particles by ID number to a
    specific galaxy, assuming each galaxy has a single SMBH in it. The list of
    masks is organised so that the first mask corresponds to particles around
    the SMBH with the lower ParticleID number.

    Parameters
    ----------
    snap: pygad <Snapshot> object
    radius: radius within which particles are assumed to belong to the host
            galaxy of the SMBH (see notes on <get_id_masks> for types)
    family: particle type we want to mask, usually 'stars'

    Returns
    -------
    masks: dict of pygad <IDMask> objects, with keys corresponding to the host
           galaxy SMBH particle ID
    """
    masks = dict()
    bh_idx_in_decreasing_id = snap.bh['ID'][np.argsort(snap.bh['ID'])]
    for i, idx in enumerate(bh_idx_in_decreasing_id):
        masks[idx] = get_id_mask(snap, idx, radius=radius, family=family)
    return masks


def get_id_mask(snap, bhid, radius=10, family='stars'):
    """
    Obtain a mask that allows filtering of particular particles.

    Parameters
    ----------
    snap: pygad <Snapshot> object
    bhid: SMBH id number we want to find particles around
    radius: radius within which particles are assumed to belong to the host
            galaxy of the SMBH; int/float for all particles in a ball, or
            [lower, upper] for all particles in a shell
    family: particle type we want to mask, usually 'stars'

    Returns
    -------
    pygad <IDMask> object to mask future snapshots of the same simulation
    """
    assert(snap._phys_units_requested)
    assert(family in ['stars', 'dm', 'bh'])
    subsnap = getattr(snap, family)
    #find the IDs of all particles close to the BH
    if isinstance(radius, int) or isinstance(radius, float):
        #option 1: in a ball
        mask = pygad.BallMask(pygad.UnitQty(radius, 'kpc'), snap.bh['pos'][snap.bh['ID']==bhid])
    elif isinstance(radius, list):
        #option 2: a shell
        assert(radius[1] > radius[0])
        outer_mask = pygad.BallMask(pygad.UnitQty(radius[1], 'kpc'), snap.bh['pos'][snap.bh['ID']==bhid])
        inner_mask = pygad.BallMask(pygad.UnitQty(radius[0], 'kpc'), snap.bh['pos'][snap.bh['ID']==bhid])
        mask = outer_mask & ~inner_mask
    ids = np.array(subsnap[mask]['ID'])
    snap.delete_blocks()
    return pygad.IDMask(ids)


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
