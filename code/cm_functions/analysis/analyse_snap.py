import warnings
import numpy as np
import scipy.linalg
import pygad

from . import masks as masks
from ..mathematics import radial_separation, density_sphere

__all__ = ['get_com_of_each_galaxy', 'get_com_velocity_of_each_galaxy', 'get_galaxy_axis_ratios', 'get_virial_info_of_each_galaxy', "calculate_Hamiltonian", "determine_if_merged", "influence_radius", "hardening_radius", "gravitational_radiation_radius", "get_G_rho_per_sigma", "shell_com_motions_each_galaxy", "projected_half_mass_radius"]


def get_com_of_each_galaxy(snap, initial_radius=10, masks=None, verbose=True, min_particle_count=10, family='stars', initial_guess=None):
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
    initial_guess: (1,3) array specifying the initial CoM guess for all 
                   galaxies, default (None) uses an initial guess of the
                   position of the BH that is associated with that galaxy

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
            print('Finding CoM associated with BH ID {:.3e}'.format(idx))
        if masks is not None:
            masked_subsnap = subsnap[masks[idx]]
        else:
            masked_subsnap = subsnap
        if initial_guess is None:
            coms[idx] = pygad.analysis.shrinking_sphere(masked_subsnap, snap.bh['pos'][snap.bh['ID']==idx, :], initial_radius, stop_N=min_particle_count)
        else:
            coms[idx] = pygad.analysis.shrinking_sphere(masked_subsnap, initial_guess, initial_radius, stop_N=min_particle_count)
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
    vcoms: dict with n keys, where keys correspond to the keys in xcom, and the 
           values correspond to the centre of mass velocity of each galaxy
    """
    assert(snap.phys_units_requested)
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    subsnap = getattr(snap, family)
    #prepare the dict that will hold the velocity centre of masses
    vcoms = dict()
    for ind, idx in enumerate(xcom.keys()):
        if masks is not None:
            masked_subsnap = subsnap[masks[idx]]
        else:
            masked_subsnap = subsnap
        #make a ball about the CoM
        ball_radius = np.sort(pygad.utils.dist(masked_subsnap['pos'], xcom[idx]))[int(min_particle_count)]
        if verbose:
            print('Maximum radius for velocity CoM set to {:.3e} kpc'.format(ball_radius))
        ball_mask = pygad.BallMask(pygad.UnitQty(ball_radius, 'kpc'), center=xcom[idx])
        vcoms[idx] = pygad.analysis.mass_weighted_mean(masked_subsnap[ball_mask], qty='vel')
    return vcoms


def get_galaxy_axis_ratios(snap, xcom=None, radial_mask=None, family='stars', return_eigenvectors=False):
    """
    Determine the axis ratios b/a and c/a of a galaxy

    Parameters
    ----------
    snap: pygad snapshot to analyse
    xcom: CoM coordinates for the galaxy
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
        vel_mag = radial_separation(snap["vel"][start:end])
        vel_mag = pygad.UnitArr(vel_mag, "km/s")
        KE += np.sum(0.5 * snap["mass"][start:end]*vel_mag**2)
        PE += np.sum(snap["pot"][start:end]*snap["mass"][start:end])
    return KE+PE


def determine_if_merged(snap):
    """
    Determine if a merger has occurred by identifying a BH with 0 mass

    Parameters
    ----------
    snap: pygad snapshot to analyse

    Returns
    -------
    merged (bool): has the bh merged?
    remnant_id: ID of the remnant BH (the one with non-zero mass)
    """
    assert snap.bh["mass"].shape[0] > 1, "The system must be a merger system!"
    merged = True if np.any(snap.bh["mass"]<1e-15) else False
    if merged:
        remnant_id = snap.bh["ID"][snap.bh["mass"]>1e-15][0]
    else:
        remnant_id = None
    return merged, remnant_id


def get_massive_bh_ID(bhs):
    """
    Determine the ID of the more massive of the two BHs

    Parameters
    ----------
    bhs: pygad bh sub-snapshot, usually of the form snap.bh

    Returns
    -------
    ID of the more massive of the two BHs
    """
    massive_idx = np.argmax(bhs["mass"])
    return bhs["ID"][massive_idx]


def influence_radius(snap, binary=True):
    """
    Determine the influence radius for the system, defined as Eq. 2.11 in
    Merritt 2013. This is denoted as r_m, whereas the alternative definition,
    Eq. 2.12, is equivalent in the case where stars have the distribution of 
    the singular isothermal sphere.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    binary (bool): should the influence radius be calculated for the binary as
                   a single object (True), or separately for each BH (False)?
    
    Returns
    -------
    r_inf (dict): keys correspond to BH ID (or the more massive BH ID if 
                  binary=True), and values to the influence radius
    """
    def _find_radius_for_mass(M, m, pos, centre):
        #determine the radius where the enclosed mass = desired mass M
        r = pygad.utils.geo.dist(pos, centre)
        r.sort()
        #determine how many m are in M -> this will be the index of r we need
        idx = int(np.ceil(M/m))-1
        return pygad.UnitScalar(r[idx], r.units)
    assert(snap.phys_units_requested)
    r_inf = dict()
    if binary:
        #we are dealing with the combined mass
        mass_bh = np.sum(snap.bh["mass"])
        centre = pygad.analysis.center_of_mass(snap.bh)
        massive_ID = get_massive_bh_ID(snap.bh)
        r = _find_radius_for_mass(2*mass_bh, snap.stars["mass"][0], snap.stars["pos"], centre=centre)
        r_inf[massive_ID] = r
    else:
        #we want the influence radius for each BH. No masking is done to separate the stars to their original galaxy
        for i in range(2):
            r = _find_radius_for_mass(2*snap.bh["mass"][i], snap.stars["mass"][0], snap.stars["pos"], snap.bh["pos"][i,:])
            r_inf[snap.bh["ID"][i]] = r
    return r_inf


def hardening_radius(bhms, rm):
    """
    Determine the hardening radius for the system, defined as Eq. 8.71 in 
    Merritt 2013. This definition uses the influence radius definition of 
    Eq. 2.11.

    Parameters
    ----------
    bhms: list of BH masses
    rm: influence radius

    Returns
    -------
    ah: hard binary radius in the same units as rm
    """
    q = bhms[0] / bhms[1]
    q = 1/q if q>1 else q #ensure q <= 1
    ah = q / (1 + q)**2 * rm / 4
    return pygad.UnitScalar(ah, rm.units)


def gravitational_radiation_radius(snap, rh, ah, tah, H, e=0):
    """
    Determine the gravitational wave radius, where da/dt due to stellar 
    interactions is equal to da/dt due to GW emission. The equation follows 
    Eq. 8.26 in Merritt 2013, however the eccentricity term may optionally be 
    included. Below, "inner" quantities typically refer to the quantity within
    the gravitational influence radius.

    Parameters
    ----------
    snap: pygad snapshot from which to estimate the inner density and velocity
          dispersion
    rh: gravitational influence radius of the binary (assumed pc)
    ah: hard binary radius (assumed pc)
    tah: time of the hard binary radius (assumed Myr)
    H: hardening rate parameter
    e: eccentricity (default 0 for a circular orbit)
    """
    assert 0 <= e <= 1
    #eccentricity function
    Q = (1-e**2)**(-3.5) * (1 + 73/24*e**2 + 37/96*e**4)
    assert(snap.phys_units_requested), "Snapshot must be given in physical units!"
    if not isinstance(rh, pygad.UnitArr):
        print("Setting rh to default length units {}".format("pc"))
        rh = pygad.UnitScalar(rh, "pc")
    rho, sigma = _get_inner_rho_and_sigma(snap, extent=rh)
    m1_m2_M = np.product(snap.bh["mass"]) * np.sum(snap.bh["mass"])
    #set the constants
    const_G = pygad.physics.G.in_units_of("pc/Msol*km**2/s**2")
    const_c = pygad.physics.c.in_units_of("km/s")
    a_5 = 64/5 * const_G**2 * m1_m2_M * sigma / (const_c**5 * rho * H) * Q
    a = pygad.UnitScalar(a_5.view(np.ndarray)**0.2, "pc")
    #time_a = pygad.UnitScalar(sigma / (const_G * rho * a), "Myr")/H
    time_a = pygad.UnitScalar(sigma/(const_G * rho) * (ah-a)/(ah*a), "Myr")/H + tah
    return a.view(np.ndarray), time_a.view(np.ndarray)


def _get_inner_rho_and_sigma(snap, extent=None):
    """
    Get the mean (3-dimensional) stellar density and velocity dispersion within 
    a given radius.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    extent: radial range within which to calculate quantities. Default of None
            uses all stars in the snapshot
    
    Returns
    -------
    inner_density: mean stellar density within a ball of radius extent
    inner_sigma: mean stellar velocity dispersion with a ball of radius extent
    """
    if extent is not None:
        assert isinstance(extent, pygad.UnitArr), "extent must have units!"
        extent_mask = pygad.BallMask(extent, center=pygad.analysis.center_of_mass(snap.bh))
        subsnap = snap.stars[extent_mask]
    else:
        warnings.warn("Inner quantities will be calculated for all stars!")
        subsnap = snap.stars
    inner_density = density_sphere(np.sum(subsnap["mass"]), extent)
    inner_sigma = np.nanmean(np.nanstd(subsnap["vel"], axis=0))
    return inner_density, inner_sigma


def get_G_rho_per_sigma(snap, extent=None):
    """
    Wrapper to determine the ratio of G*rho/sigma. 

    Parameters
    ----------
    snap: pygad snapshot to analyse
    extent: radial range within which to calculate quantities. Default of None
            uses all stars in the snapshot
    
    Returns
    -------
    G*rho/sigma within radius 'extent', in units of pc^-1 yr^-1
    """
    inner_density, inner_sigma = _get_inner_rho_and_sigma(snap, extent)
    G_rho_per_sigma = pygad.physics.G * inner_density / inner_sigma
    return G_rho_per_sigma.in_units_of("pc**-1/yr")


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
    global_vcom = get_com_velocity_of_each_galaxy(snap, global_xcom, masks=id_masks, family=family, verbose=verbose)
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
    return xcoms, vcoms, global_xcom, global_vcom


def projected_half_mass_radius(snap, obs=10, family="stars", masks=None):
    """
    Determine the projected half mass radius of (potentially two) galaxies

    Parameters
    ----------
    snap: pygad snapshot to analyse
    obs: number of random rotations to perform
    family: determine the projected half mass radius for this particle family
    masks: dict of id masks (from masks.get_all_id_masks) so that the half mass
           radius can be determined for two galaxies within a merger simulation.
           If None, the system is treated as a whole and the projected half
           mass radius is assigned to just one BH ID.
    
    Returns
    -------
    Re: dict of half mass radii of systems, with keys correspponding to the BH
        ID associated with the galaxy
    """
    assert(snap.phys_units_requested)
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    Re = dict.fromkeys(snap.bh["ID"], 0)
    rng = np.random.default_rng()
    rot_axis = rng.uniform(-1, 1, (obs, 3))
    rot_angle = rng.uniform(0, np.pi, obs)
    for j, bhid in enumerate(Re.keys()):
        if masks is None:
            centre_guess = pygad.analysis.center_of_mass(snap.bh)
            subsnap = getattr(snap, family)
            if j > 0:
                break
        else:
            centre_guess = snap.bh[snap.bh["ID"]==bhid]["pos"]
            subsnap = snap[masks[bhid]]
        for i in range(obs):
            rot = pygad.transformation.rot_from_axis_angle(rot_axis[i], rot_angle[i])
            rot.apply(subsnap)
            centre = pygad.analysis.shrinking_sphere(subsnap, centre_guess, 10)
            for proj in range(3):
                Re[bhid] += pygad.analysis.half_mass_radius(subsnap, center=centre, proj=proj)
        Re[bhid] /= (obs*3)
    return Re

