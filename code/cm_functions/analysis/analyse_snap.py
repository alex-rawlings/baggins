import warnings
import numpy as np
import scipy.linalg, scipy.interpolate
import pygad

from . import masks as masks
from ..mathematics import radial_separation, density_sphere
from .general import snap_num_for_time
from ..general import convert_gadget_time, set_seed_time


__all__ = ['get_com_of_each_galaxy', 'get_com_velocity_of_each_galaxy', 'get_galaxy_axis_ratios', 'get_virial_info_of_each_galaxy', "calculate_Hamiltonian", "determine_if_merged", "enclosed_mass_radius", "influence_radius", "hardening_radius", "gravitational_radiation_radius", "get_inner_rho_and_sigma", "get_G_rho_per_sigma", "shell_com_motions_each_galaxy", "projected_quantities", "inner_DM_fraction", "shell_flow_velocities", "angular_momentum_difference_gal_BH", "loss_cone_angular_momentum"]


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
    eigen_vals = np.real(eigen_vals[sorted_idx])
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


def enclosed_mass_radius(snap, binary=True, mass_frac=1):
    """
    Determine the radius containining the given mass.

    Parameters
    ----------
    snap: pygad snapshot to analyse
    binary (bool): should the influence radius be calculated for the binary as
                   a single object (True), or separately for each BH (False)?
    mass_frac: fraction of the mass to search for. Influence radius corresponds
               to mass_frac = 2.
    
    Returns
    -------
    r (dict): keys correspond to BH ID (or the more massive BH ID if 
                  binary=True), and values to the influence radius
    """
    def _find_radius_for_mass(M, m, pos, centre):
        #determine the radius where the enclosed mass = desired mass M
        r = pygad.utils.geo.dist(pos, centre)
        r.sort()
        #interpolate in mass-radius plane
        #determine how many m are in M -> this will be the index of r we need
        idx = int(np.ceil(M/m))-1
        ms = np.array([idx, idx+1])*m
        f = scipy.interpolate.interp1d(ms, [r[idx], r[idx+1]])
        return pygad.UnitScalar(f(M), r.units)
    assert(snap.phys_units_requested)
    r = dict()
    if binary:
        #we are dealing with the combined mass
        mass_bh = np.sum(snap.bh["mass"])
        centre = pygad.analysis.center_of_mass(snap.bh)
        massive_ID = get_massive_bh_ID(snap.bh)
        _r = _find_radius_for_mass(mass_frac*mass_bh, snap.stars["mass"][0], snap.stars["pos"], centre=centre)
        r[massive_ID] = _r
    else:
        #we want the influence radius for each BH. No masking is done to separate the stars to their original galaxy
        bhids = snap.bh["ID"]
        bhids.sort()
        for id in bhids:
            bh_id_mask = pygad.IDMask(id)
            _r = _find_radius_for_mass(mass_frac*snap.bh[bh_id_mask]["mass"][0], snap.stars["mass"][0], snap.stars["pos"], centre=snap.bh[bh_id_mask]["pos"][0])
            r[id] = _r
    return r


def influence_radius(snap, binary=True):
    """
    Determine the influence radius for the system, defined as Eq. 2.11 in
    Merritt 2013. This is denoted as r_m, whereas the alternative definition,
    Eq. 2.12, is equivalent in the case where stars have the distribution of 
    the singular isothermal sphere.
    This is a wrapper around the more general method enclosed_mass_radius().

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
    return enclosed_mass_radius(snap, binary, mass_frac=2)


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


def gravitational_radiation_radius(bh_masses, ah, tah, H, Gps, e=0):
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

    Returns
    -------
    a: gravitational wave radius in pc
    time_a: time in Myr when the binary is expected to have a = a_GR
    """
    assert 0 <= e <= 1
    #eccentricity function
    Q = (1-e**2)**(-3.5) * (1 + 73/24*e**2 + 37/96*e**4)
    m1_m2_M = np.product(bh_masses) * np.sum(bh_masses)
    #set the constants
    const_G = pygad.physics.G.in_units_of("pc/Msol*km**2/s**2")
    const_c = pygad.physics.c.in_units_of("km/s")
    a_5 = 64/5 * const_G**3 * m1_m2_M  / (const_c**5 * Gps * H) * Q
    a = pygad.UnitScalar(a_5.in_units_of("pc**5").view(np.ndarray)**0.2, "pc")
    time_a = pygad.UnitScalar((ah-a)/(ah*a) / Gps, "Myr")/H + tah
    return a.view(np.ndarray), time_a.view(np.ndarray)


def get_inner_rho_and_sigma(snap, extent=None):
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


def get_G_rho_per_sigma(snaplist, t, extent=None):
    """
    Wrapper to determine the ratio of G*rho/sigma. The value G*rho/sigma is
    calculated for the snapshot before and the snapshot after the desired time
    t, and linear interpolation performed to get the value at t.

    Parameters
    ----------
    snaplist: list of snapshots from which to determine the quantity
    t: time to determine the quantity for (Myr)
    extent: radial range within which to calculate quantities. Default of None
            uses all stars in the snapshot
    
    Returns
    -------
    G*rho/sigma within radius 'extent', in units of pc^-1 yr^-1
    """
    ts = np.full((2), np.nan)
    inner_density = np.full_like(ts, np.nan)
    inner_sigma = np.full_like(ts, np.nan)
    idx = snap_num_for_time(snaplist, t, method="floor", units="Myr")
    for i in range(2):
        snap = pygad.Snapshot(snaplist[idx+i], physical=True)
        ts[i] = convert_gadget_time(snap, new_unit="Myr")
        rho_temp, sigma_temp = get_inner_rho_and_sigma(snap, extent)
        rho_units = rho_temp.units
        sigma_units = sigma_temp.units
        inner_density[i], inner_sigma[i] = rho_temp, sigma_temp
    f_rho = scipy.interpolate.interp1d(ts, inner_density)
    f_sigma = scipy.interpolate.interp1d(ts, inner_sigma)
    G_rho_per_sigma = pygad.physics.G * pygad.UnitScalar(f_rho(t), rho_units) / pygad.UnitScalar(f_sigma(t), sigma_units)
    return G_rho_per_sigma.in_units_of("pc**-1/yr")


def shell_com_motions_each_galaxy(snap, separate_galaxies=True, shell_kw={"start":1e-6, "stop":500, "num":20}, family="stars", Gcom_kw={"initial_radius":10, "min_particle_count":10}, verbose=True):
    """
    Determine the CoM motions within concentric shells, as opposed 
    to a global CoM motion value

    Parameters
    ----------
    snap: pygad snapshot to analyse
    separate_galaxies: bool, apply a particle ID mask to separate the galaxies
    shell_kw: dict to pass np.geomspace controlling the shell properties:
                  start: inner radius of inner shell
                  stop: outer radius of outer shell
                  num: number of shells
    family: pygad particle family to do the analysis for
    Gcom_kw: dict of optional arguments to pass to get_com_of_each_galaxy, this
             determines the "global" CoM properties of each galaxy
    verbose: bool, verbose printing?

    Returns
    -------
    xcoms: dict, CoM position of each shell
    vcoms: dict, CoM velocity of each shell
    global_xcom: dict, global CoM position of each galaxy
    global_vcom: dict, global CoM velocity of each galaxy
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


def projected_quantities(snap, obs=10, family="stars", masks=None, q=[0.25, 0.75], r_edges=np.geomspace(2e-1, 20, 51)):
    """
    Determine projected quantities of:
        - half mass radius,
        - velocity dispersion
    of (potentially two) galaxies

    Parameters
    ----------
    snap: pygad snapshot to analyse
    obs: number of random rotations to perform
    family: determine the projected half mass radius for this particle family
    masks: dict of id masks (from masks.get_all_id_masks) so that the half mass
           radius can be determined for two galaxies within a merger simulation.
           If None, the system is treated as a whole and the projected half
           mass radius is assigned to just one BH ID.
    q: lower and upper quantiles for error bounds (more robust that std)
    r_edges: edges of radial bins for density profile
    
    Returns
    -------
    Q: dict of dicts, with level 1 keys corresponding to the BH ID associated 
       with the galaxy, and level 2 keys corresponding to the quantity
    """
    assert(snap.phys_units_requested)
    q.sort()
    num_bhs = len(snap.bh['mass'])
    if masks is not None:
        assert(len(masks) == num_bhs)
    #pre-allocate dictionaries
    Re = dict.fromkeys(snap.bh["ID"], {"estimate":0, "low":0, "high":0})
    vsig = dict.fromkeys(snap.bh["ID"], {"estimate":0, "low":0, "high":0})
    rho = dict.fromkeys(snap.bh["ID"], {"estimate":0, "low":0, "high":0})
    #set up rng and distributions 
    rng = np.random.default_rng(set_seed_time())
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
        #temporary arrays to store data
        Re_temp = np.full(obs*3, np.nan)
        vvar_temp = np.full(obs*3, np.nan)
        rho_temp =  np.full((obs*3, len(r_edges)-1), np.nan)
        for i in range(obs):
            rot = pygad.transformation.rot_from_axis_angle(rot_axis[i], rot_angle[i])
            rot.apply(subsnap)
            centre = pygad.analysis.shrinking_sphere(subsnap, centre_guess, 10)
            for proj in range(3):
                Re_temp[3*i+proj] = pygad.analysis.half_mass_radius(subsnap, center=centre, proj=proj)
                #we want vel dispersion within Re
                ball_mask = pygad.BallMask(Re_temp[3*i+proj], center=centre)
                vvar_temp[3*i+proj] = pygad.analysis.los_velocity_dispersion(subsnap[ball_mask], proj=proj)**2
                rho_temp[3*i+proj, :] = pygad.analysis.profile_dens(subsnap, qty="mass", r_edges=r_edges, center=centre)
        subsnap.delete_blocks()
        Re[bhid]["estimate"] = np.nanmedian(Re_temp)
        Re[bhid]["low"] = np.nanquantile(Re_temp, q[0])
        Re[bhid]["high"] = np.nanquantile(Re_temp, q[1])
        vsig[bhid]["estimate"] = np.sqrt(np.nanmedian(vvar_temp))
        vsig[bhid]["low"] = np.sqrt(np.nanquantile(vvar_temp, q[0]))
        vsig[bhid]["high"] = np.sqrt(np.nanquantile(vvar_temp, q[1]))
        rho[bhid]["estimate"] = np.nanmedian(rho_temp, axis=0)
        rho[bhid]["low"] = np.nanquantile(rho_temp, q[0], axis=0)
        rho[bhid]["high"] = np.nanquantile(rho_temp, q[1], axis=0)
        if masks is None:
            break
    return Re, vsig, rho


def inner_DM_fraction(snap, Re=None):
    """
    Determine the dark matter fraction within 1 Re

    Parameters
    ----------
    snap: pygad snapshot to analyse
    Re: effective radius. Default (None) calculates the value

    Returns
    -------
    fraction of DM within 1 Re
    """
    if Re is None:
        Re,*_ = projected_quantities(snap)
        Re = list(Re.values())[0]["estimate"]
    centre_guess = pygad.analysis.center_of_mass(snap.bh)
    ball_mask = pygad.BallMask(Re, center=centre_guess)
    dm_mass = snap.dm["mass"][0]
    star_mass = snap.stars["mass"][0]
    return len(snap.dm[ball_mask])*dm_mass / (len(snap.dm[ball_mask])*dm_mass + len(snap.stars[ball_mask])*star_mass)


def shell_flow_velocities(snap, R, centre=None, direction="out", dt="5 Myr"):
    """
    Return the velocities of those particles moving either inwards or outwards
    through a shell of radius R. This function is largely based on the 
    implementation in pygad.analysis.flow_rates().
    Note that NO CENTRING IS PERFORMED.

    Parameters
    ----------
    snap: pygad snapshot to use
    R: shell radius, assumed to be in the same units as the position units of 
       snap
    direction: either "in" or "out", which direction the particles are moving in
    dt: time used for the linear extrapolation of current positions (default 
        units of Myr)
    
    Returns
    -------
    array of radial velocities corresponding to those particles moving in the 
    desired direction through a shell with radius R.
    """
    assert direction in ["in", "out"], "Flow direction must be either in or out"
    R = pygad.UnitScalar(R, units=snap["r"].units)
    dt = pygad.UnitScalar(dt, units="Myr")
    #this step is to prevent the conversion of an entire array
    dt.convert_to(snap["r"].units / snap["vel"].units)
    if centre is None:
        centre = [0.,0.,0.]
    centre = pygad.UnitArr(centre, units=snap["r"].units)
    t = pygad.Translation(-centre)
    # radial velocity appears to be updated after translation too
    t.apply(snap)
    rpred = snap["r"] + snap["vrad"] * dt
    if direction == "out":
        #particles are less than R originally, but move outwards
        mask = (snap["r"] < R) & (rpred >= R)
    else:
        #particles are further than R originally, but move inwards
        mask = (snap["r"] >= R) & (rpred < R)
    return snap[mask]["vrad"].view(np.ndarray)


def angular_momentum_difference_gal_BH(snap):
    """
    Determine the angular difference between the angular momentum of the stellar
    component of a galaxy and the BHs. Theta is defined as in Nasim et al. 2021
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..498N/abstract 

    Parameters
    ----------
    snap: pygad snapshot to use

    Returns
    -------
    theta: angle between L_gal and L_bh
    """
    assert snap.phys_units_requested
    L_gal = snap.stars["angmom"].sum(axis=0)
    L_bh = snap.bh["angmom"].sum(axis=0)
    theta = np.arccos(np.dot(L_gal, L_bh) / (radial_separation(L_gal) * radial_separation(L_bh)))
    return theta


def loss_cone_angular_momentum(snap, a, kappa=1):
    """
    Calculate the approximate angular momentum of the loss cone, as from 
    Gualandris et al. 2017, but multiplied by the stellar mass 
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.2301G/abstract 

    Parameters
    ----------
    snap: pygad snapshot to use
    kappa: dimensionless constant

    Returns
    -------
    Loss cone ang. mom.
    """
    assert snap.phys_units_requested
    J_unit = snap["angmom"].units
    starmass = pygad.UnitScalar(snap.stars["mass"][0], snap.stars["mass"].units)
    Mbin = snap.bh["mass"].sum()
    const_G = const_G = pygad.physics.G.in_units_of("pc/Msol*km**2/s**2")
    J = np.sqrt(2 * const_G * Mbin * kappa * a) * starmass
    return J.in_units_of(J_unit)
