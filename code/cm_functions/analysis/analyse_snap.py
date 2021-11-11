import numpy as np
import scipy.linalg
import pygad

from . import masks as masks
from . import orbit as orbit

__all__ = ['get_com_of_each_galaxy', 'get_com_velocity_of_each_galaxy', 'get_galaxy_axis_ratios', 'get_virial_info_of_each_galaxy', "calculate_Hamiltonian"]


def get_com_of_each_galaxy(snap, initial_radius=10, masks=None, verbose=True, min_particle_count=10, family='stars'):
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
    family: particle family to analyse

    Returns
    -------
    coms: dict with n keys, where each key corresponds to the centre of mass
          of each galaxy
    """
    assert(snap.phys_units_requested)
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    subsnap = getattr(snap, family)
    #prepare dict that will hold the centre of masses
    coms = dict()
    for ind, idx in enumerate(snap.bh['ID']):
        if verbose:
            print('Finding CoM by centring on BH {}'.format(ind))
        if masks is not None:
            masked_subsnap = subsnap[masks[idx]]
        else:
            masked_subsnap = subsnap
        coms[idx] = pygad.analysis.shrinking_sphere(masked_subsnap, snap.bh['pos'][snap.bh['ID']==idx, :], initial_radius, stop_N=min_particle_count)
    return coms


def get_com_velocity_of_each_galaxy(snap, xcom, masks=None, min_particle_count=5e4, verbose=True, family='stars'):
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
    assert(snap.phys_units_requested)
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    subsnap = getattr(snap, family)
    #prepare the dict that will hold the velocity centre of masses
    vcoms = dict()
    for ind, idx in enumerate(snap.bh['ID']):
        if masks is not None:
            masked_subsnap = subsnap[masks[idx]]
        else:
            masked_subsnap = subsnap
        #make a ball about the CoM
        ball_radius = np.sort(pygad.utils.dist(masked_subsnap['pos'], xcom[idx]))[int(min_particle_count)]
        if verbose:
            print('Maximum radius for velocity CoM set to {} kpc'.format(ball_radius))
        ball_mask = pygad.BallMask(pygad.UnitQty(ball_radius, 'kpc'), center=xcom[idx])
        vcoms[idx] = pygad.analysis.mass_weighted_mean(masked_subsnap[ball_mask], qty='vel')
    return vcoms


def get_galaxy_axis_ratios(snap, xcom=None, radial_mask=None, family='stars', return_eigenvectors=False):
    """
    Determine the axis ratios b/a and c/a of a galaxy

    Parameters
    ----------
    snap: pygad snapshot to analyse
    xcom: dict of CoM coordinates for each galaxy, assumes the dict keys are the
          BH particle IDs
    masks: pygad radial masks to apply to the (sub) snapshot
    family: particle family to analyse
    return_eigenvectors: bool, true to return eigenvectors as well as axis 
                         ratios
    
    Returns
    -------
    axis ratios in order b/a, c/a
    optionally return eigenvectors corresponding to a, b, c
    """
    if xcom is None:
        xcom = pygad.UnitArr([0,0,0], 'kpc')
    elif not isinstance(xcom, pygad.UnitArr):
        xcom = pygad.UnitArr(xcom, 'kpc')
    subsnap = getattr(snap, family)
    #move entire subsnap to CoM coordinates
    #just doing this for a masked section results in the radial distance 'r'
    #block not being rederived...
    pygad.Translation(-xcom).apply(subsnap)
    if radial_mask is not None:
        #apply either a ball or shell mask
        subsnap = subsnap[radial_mask]
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


def get_virial_info_of_each_galaxy(snap, xcom=None, masks=None):
    """
    Extract the virial mass and radius from either the initial set up of a 
    merger system, or when the merger has relaxed.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    xcom: dict of CoM coordinates for each galaxy, assumes the dict keys are the
          BH particle IDs
    masks: particle ID masks. Can be either a dict (thus only one particle 
           family goes into the calculation), or a list of dicts (thus stars 
           and dm go into the calculation)

    Returns
    -------
    virial_radius: virial radius of system. If masks were used, this is a dict 
                   with the keys being the BH IDs and the mass corresponding to 
                   that galaxy as values
    virial_mass: virial mass of system. Same output style as virial_radius
    """
    if masks is None:
        return pygad.analysis.virial_info(snap, center=xcom)
    elif isinstance(masks, list):
        mask_list = masks
        masks = dict()
        for key in mask_list[0].keys():
            masks[key] = mask_list[0][key] | mask_list[1][key]
    elif not isinstance(masks, dict):
        raise ValueError("masks must be a list of dicts, a single dict, or None!")
    virial_mass = dict()
    virial_radius = dict()
    for key, this_com in xcom.items():
        virial_radius[key], virial_mass[key] = pygad.analysis.virial_info(snap[masks[key]], center=this_com)
    return virial_radius, virial_mass


def calculate_Hamiltonian(snap, chunk=1e5):
    """
    Determine the total Hamiltonian of a system. Requires that ketju has been
    compiled with the:
        OUTPUTPOTENTIAL
    flag (so that the potential is saved to the snapshots)

    Parameters
    ----------
    snap: pygad snapshot to analyse
    chunk: perform summation in chunks of this size for efficiency

    Returns
    -------
    total energy (the Hamiltonian)
    """
    chunk = int(chunk)
    total_N = snap["pos"].shape[0]
    KE = 0
    PE = 0
    for start in range(0, total_N, chunk):
        end = min(start+chunk, total_N)
        vel_mag = orbit.radial_separation(snap["vel"][start:end])
        vel_mag = pygad.UnitArr(vel_mag, "km/s")
        KE += np.sum(0.5 * snap["mass"][start:end]*vel_mag**2)
        PE += np.sum(snap["pot"][start:end]*snap["mass"][start:end])
    return KE+PE


############################################
########## DISCONTINUED FUNCTIONS ##########
############################################

def shell_com_motions_each_galaxy(snap, separate_galaxies=True, shell_kw={"start":1e-6, "stop":500, "num":20}, family="stars", Gcom_kw={"initial_radius":10, "min_particle_count":10}, verbose=True):
    """
    Determine the CoM motions within concentric shells, as opposed 
    to a global CoM motion value

    Parameters
    ----------
    snap: pygad snapshot to analyse
    family: particle family to analyse

    Returns
    -------

    """
    if separate_galaxies:
        #mask the particles as belonging to one of two progenitors
        id_masks = masks.get_all_id_masks(snap, family=family)
    else:
        subsnap = getattr(snap, family)
        id_masks = {snap.bh["ID"][0]: pygad.IDMask(subsnap["ID"])}
    xcoms = dict()
    vcoms = dict()
    for k in id_masks.keys():
        xcoms[k] = np.full((shell_kw["num"], 3), np.nan)
        vcoms[k] = np.full((shell_kw["num"], 3), np.nan)
    global_xcom = get_com_of_each_galaxy(snap, family=family, verbose=verbose, masks=id_masks, **Gcom_kw)
    shell_radii = np.geomspace(**shell_kw)
    #iterate over each shell
    for i, (r_inner, r_outer) in enumerate(zip(
        shell_radii[:-1], shell_radii[1:]
    )):
        #mask particles to this shell
        radial_mask = masks.get_all_radial_masks(snap, (r_inner, r_outer), centre=global_xcom, id_masks=id_masks, family=family)
        #compute CoM motions for shell using mass-weighted means
        for bhid in id_masks.keys():
            xcoms[bhid][i, :] = pygad.analysis.mass_weighted_mean(snap[radial_mask[bhid]], qty="pos")
            vcoms[bhid][i, :] = pygad.analysis.mass_weighted_mean(snap[radial_mask[bhid]], qty="vel")
    #we now have datasets with CoM motion as a function of radius
    #interpolate the results