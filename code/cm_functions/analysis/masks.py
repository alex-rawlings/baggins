import numpy as np
import pygad

__all__ = ["get_id_mask", "get_all_id_masks", "get_radial_mask", "get_all_radial_masks", "get_binding_energy_mask"]

def get_id_mask(snap, bhid, family='stars'):
    """
    Obtain a mask that allows filtering of particular particles. Particles are
    first filtered to the host galaxy depending on their particle ID number, 
    where it is necessarily assumed that the particle ID ordering follows:
        galaxy 1 IDs --> galaxy 2 IDs
    To obtain a mask of all particles in a system, the routine can be run 
    for each particle family, and the masks combined as e.g.:
        all_id_mask = star_id_mask & dm_id_mask & bh_id_mask
    #TODO should this be a separate method??

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
    #get all the particles in the family -> less sorting to do
    subsnap = getattr(snap, family)
    subsnap_ids = np.sort(subsnap['ID'])
    #now need all particles that belong to the galaxy of the desired SMBH
    break_idx = np.where(np.diff(subsnap_ids)>1)[0]
    if len(break_idx) > 1:
        raise RuntimeError('Particles of one family expected to be organised in two continuous groups!')
    elif len(break_idx) < 1:
        raise RuntimeError('There is only one galaxy in the system!')
    break_idx = break_idx[0]+1
    if bhid == min(snap.bh['ID']):
        ids = np.array(subsnap[subsnap['ID'] < subsnap_ids[break_idx]]['ID'])
    else:
        ids = np.array(subsnap[subsnap['ID'] >= subsnap_ids[break_idx]]['ID'])
    snap.delete_blocks()
    return pygad.IDMask(ids)


def get_all_id_masks(snap, family='stars'):
    """
    Return a dict of masks that mask the chosen particles by ID number to a
    specific galaxy, assuming each galaxy has a single SMBH in it.

    Parameters
    ----------
    snap: pygad <Snapshot> object
    family: particle type we want to mask, usually 'stars'

    Returns
    -------
    masks: dict of pygad <IDMask> objects, with keys corresponding to the host
           galaxy SMBH particle ID
    """
    masks = dict()
    for i, idx in enumerate(snap.bh['ID']):
        masks[idx] = get_id_mask(snap, idx, family=family)
    return masks


def get_radial_mask(snap, radius, centre=None, id_mask=None, family=None):
    """
    Create a radial-based mask, can be either a ball or a shell. The default
    function arguments are designed for compatability with get_all_radial_masks.

    Parameters
    ----------
    snap: pygad <Snapshot> object to create the mask for
    radius: the radius to constrain the particles to - can be either a number
            to construct a ball mask, or a tuple of (inner_radius, outer_radius)
            to construct a shell mask
    centre: the centre from which the radial measurements should be made
    id_masks: dict of ID masks to constrain particles to a given galaxy
    family: particle type to construct the mask for

    Returns
    -------
    mask: pygad mask object
    """
    assert(snap.phys_units_requested)
    if family is not None:
        assert(family in ['stars', 'dm', 'bh'])
        snap = getattr(snap, family)
    if centre is None:
        centre = pygad.UnitArr([0,0,0], 'kpc')
    if isinstance(radius, int) or isinstance(radius, float):
        #option 1: in a ball
        mask = pygad.BallMask(pygad.UnitQty(radius, 'kpc'), centre)
    elif isinstance(radius, tuple):
        #option 2: a shell
        assert(radius[1] > radius[0] and len(radius)==2)
        outer_mask = pygad.BallMask(pygad.UnitQty(radius[1], 'kpc'), centre)
        inner_mask = pygad.BallMask(pygad.UnitQty(radius[0], 'kpc'), centre)
        mask = outer_mask & ~inner_mask
    else:
        raise ValueError('Radius must be either a number (ball mask) or a tuple (shell mask)!')
    if id_mask is not None:
        mask = mask & id_mask
    snap.delete_blocks()
    return mask


def get_all_radial_masks(snap, radius, centre=None, id_masks=None, family='stars'):
    """
    Return a dict of masks that mask the chosen particles by radial distance to
    the desired centre. Either a ball mask or a shell mask may be constructed.
    Additionally, the option to initially filter particles by ID to constrain
    the mask to those particles belonging to a given galaxy is possible.

    Parameters
    ----------
    snap: pygad <Snapshot> object to create the mask for
    radius: the radius to constrain the particles to - can be either a number
            to construct a ball mask, or a tuple of (inner_radius, outer_radius)
            to construct a shell mask
    centre: the centre from which the radial measurements should be made
            may be None (corresponds to [0,0,0]), a dict of coordinate arrays (e.g. the CoM) with keys corresponding to the BH ids, or "bh" to  
            centre on a BH
    id_masks: dict of ID masks to constrain particles to a given galaxy
    family: particle type to construct the mask for

    Returns
    -------
    masks: dict of pygad mask objects, with keys corresponding to the host 
           galaxy SMBH particle ID number
    """
    assert(snap._phys_units_requested)
    assert(isinstance(family, str) and family in ['stars', 'dm', 'bh'])
    #restrict snap to a particle type so we're dealing with smaller objects
    subsnap = getattr(snap, family)
    if centre is None:
        #default: set centre to origin
        centre = dict()
        for idx in snap.bh['ID']:
            centre[idx] = pygad.UnitArr([0,0,0], 'kpc')
    elif isinstance(centre, dict):
        #radial position defined from CoM dict, where keys are the BH ID
        for k in centre.keys():
            assert(k in snap.bh['ID'])
    elif centre == 'bh':
        #radial position defined from the BH
        centre = dict()
        for idx in snap.bh['ID']:
            centre[idx] = snap.bh['pos'][snap.bh['ID']==idx, :]
    else:
        raise ValueError('parameter <centre> must be one of None, dict, or "bh"!')
    if id_masks is None:
        #select all particles of the subsnap
        id_masks = dict()
        for idx in snap.bh['ID']:
            id_masks[idx] = pygad.IDMask(subsnap['ID'])
    masks = dict()
    for i, idx in enumerate(centre.keys()):
        masks[idx] = get_radial_mask(subsnap, radius=radius, centre=centre[idx], id_mask=id_masks[idx])
    snap.delete_blocks()
    return masks


def get_binding_energy_mask(snap, energy=None, id_mask=None, family=None):
    """
    Mask particles into bins of binding energy. Assumes that we are in centre of mass coordinates (including velocity): no centring is done.

    Parameters
    ----------
    snap: pygad <Snapshot> object to create the mask for
    energy: energy bins to mask particles into -  can either be a regular 
            array, in which a "ball" mask is created, or a list of tuples, in 
            which a "shell" mask is created. The default None creates a "shell" 
            type mask from the 5% to 95% quantile of binding energy in 20 bins 
            evenly spaced in logscale
    id_mask: ID masks to constrain particles to a given galaxy
    family: apply the masking to only this family (default None uses all types)

    Returns
    -------
    generator which returns the mask for that energy interval, the energy 
    sequence, and the energy units
    """
    assert(snap.phys_units_requested)
    if family is not None:
        assert(family in ["stars", "dm", "bh"])
        snap = getattr(snap, family)
    # determine binding energy of each particle
    binding_energy = -snap["mass"] * snap["pot"] - snap["mass"] * pygad.utils.geo.dist(snap["vel"])**2
    #set up default energy binning
    if energy is None:
        lowerq = np.nanquantile(binding_energy, 0.05)
        upperq = np.nanquantile(binding_energy, 0.95)
        temp_e = np.geomspace(lowerq, upperq, 21)
        energy = list(zip(temp_e[:-1], temp_e[1:]))
    # now determine if we use a ball or a shell
    if isinstance(energy[0], (int, float)):
        # option 1: ball
        for e in energy:
            bool_mask = binding_energy < e
            mask = pygad.IDMask(snap["ID"][bool_mask])
            if id_mask is not None:
                mask = mask & id_mask
            yield (mask, energy, str(binding_energy.units).strip("[]"))
    elif isinstance(energy[0], (list, tuple)):
        # option 2: shell
        for e1e2 in energy:
            assert e1e2[0] < e1e2[1], f"Lower energy bound ({e1e2[0]:.3e}) must be less than upper energy bound{e1e2[1]:.3e}!"
            bool_mask = np.logical_and(binding_energy>=e1e2[0], binding_energy<e1e2[1])
            mask = pygad.IDMask(snap["ID"][bool_mask])
            if id_mask is not None:
                mask = mask & id_mask
            yield (mask, energy, str(binding_energy.units).strip("[]"))
    else:
        raise ValueError("energy must either be an array, or a list of tuple-like objects")

