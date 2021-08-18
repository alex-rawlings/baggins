import numpy as np
import pygad

__all__ = ['get_coms_of_each_galaxy', 'get_all_id_masks', 'get_id_mask']


def get_coms_of_each_galaxy(snap, initial_radius=10, masks=None, verbose=True):
    """
    Determine the centre of mass of each galaxy in the simulation, assuming each
    galaxy has a single SMBH near its centre.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    initial_radius: initial radius for shrinking sphere method
    verbose: print verbose output

    Returns
    -------
    coms: array of shape (n,3) where the rows correspond to the centre of mass
          of each galaxy
    """
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    #prepare array that will hold the centre of masses
    coms = np.full((num_bhs, 3), np.nan)
    if masks is not None:
        for ind, idx in enumerate(list(masks.keys())):
            if verbose:
                print('Finding CoM of galaxy {}'.format(ind))
            coms[ind, :] = pygad.analysis.shrinking_sphere(snap.stars[masks[idx]], snap.bh['pos'][snap.bh['ID']==idx, :], initial_radius)
    else:
        for ind in range(num_bhs):
            if verbose:
                print('Finding CoM of galaxy {}'.format(ind))
            coms[ind, :] = pygad.analysis.shrinking_sphere(snap.stars, snap.bh['pos'][ind, :], initial_radius)
    return coms


def get_all_id_masks(snap, radius=20, family='stars'):
    """
    Return a list of masks that mask the chosen particles by ID number to a
    specific galaxy, assuming each galaxy has a single SMBH in it. The list of
    masks is organised so that the first mask corresponds to particles around
    the more massive SMBH.

    Parameters
    ----------
    snap: pygad <Snapshot> object
    radius: radius within which particles are assumed to belong to the host
            galaxy of the SMBH
    family: particle type we want to mask, usually 'stars'

    Returns
    -------
    masks: dict of pygad <IDMask> objects
    """
    masks = dict()
    bh_idx_in_decreasing_mass = snap.bh['ID'][np.argsort(snap.bh['mass'])]
    for i, idx in enumerate(bh_idx_in_decreasing_mass):
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
            galaxy of the SMBH
    family: particle type we want to mask, usually 'stars'

    Returns
    -------
    pygad <IDMask> object to mask future snapshots of the same simulation
    """
    assert(snap._phys_units_requested)
    assert(family in ['stars', 'dm', 'bh'])
    subsnap = getattr(snap, family)
    #find the IDs of all particles close to the BH
    mask = pygad.BallMask(pygad.UnitQty(radius, 'kpc'), snap.bh['pos'][snap.bh['ID']==bhid])
    ids = np.array(subsnap[mask]['ID'])
    snap.delete_blocks()
    return pygad.IDMask(ids)
