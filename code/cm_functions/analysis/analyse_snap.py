import warnings
import numpy as np
import scipy.linalg, scipy.interpolate
import pygad

from . import masks as masks
from ..mathematics import radial_separation, density_sphere
from .general import snap_num_for_time
from ..general import convert_gadget_time, set_seed_time, unit_as_str


__all__ = ['get_com_of_each_galaxy', 'get_com_velocity_of_each_galaxy', 'get_galaxy_axis_ratios', 'get_virial_info_of_each_galaxy', "virial_ratio", "calculate_Hamiltonian", "determine_if_merged", "get_massive_bh_ID", "enclosed_mass_radius", "influence_radius", "hardening_radius", "gravitational_radiation_radius", "get_inner_rho_and_sigma", "get_G_rho_per_sigma", "shell_com_motions_each_galaxy", "projected_quantities", "inner_DM_fraction", "shell_flow_velocities", "angular_momentum_difference_gal_BH", "loss_cone_angular_momentum", "escape_velocity", "count_new_hypervelocity_particles"]


def get_com_of_each_galaxy(snap, method="pot", masks=None, verbose=True, family="all", initial_radius=20):
    """
    Determine the centre of mass of each galaxy in the simulation, assuming each
    galaxy has a single SMBH near its centre.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    method : str, optional
        use minimum potential method (pot) or shrinking sphere method (ss), by 
        default "pot"
    masks : dict, optional
        pygad masks to apply to the (sub) snapshot, by default None
    verbose : bool, optional
        print verbose output, by default True
    family : str, optional
        particle family to analyse, by default "all"
    initial_radius : float, optional
        initial radius guess for shrinking sphere [kpc], by default 20

    Returns
    -------
    coms: dict
        dict with keys of bh ids, where each key corresponds to the centre of 
          mass of each galaxy
    """
    assert(snap.phys_units_requested)
    assert(method in ["pot", "ss"])
    num_bhs = len(snap.bh)
    def _yield_masked_subsnap(s=snap, masks=masks, family=family):
        # helper function to get the maybe masked-, maybe sub-, snapshot
        if masks is None:
            if family=="all":
                for i in range(num_bhs): yield (s, snap.bh["ID"][i])
            else:
                for i in range(num_bhs): yield (getattr(s, family), snap.bh["ID"][i])
        else:
            assert(len(masks) == num_bhs)
            for id, m in masks.items():
                if family=="all":
                    yield (s[m], id)
                else:
                    ss = getattr(s, family)
                    yield (ss[m], id)
    
    coms = dict.fromkeys(snap.bh["ID"], None)
    masked_subsnap_gen = _yield_masked_subsnap()
    if method == "pot":
        for i in range(num_bhs):
            masked_subsnap, bhid = next(masked_subsnap_gen)
            if masks is None:
                if i==0:
                    min_pot_idx = np.argmin(masked_subsnap["pot"])
                    coms[bhid] = masked_subsnap["pos"][min_pot_idx, :]
                else:
                    coms[bhid] = list(coms.values())[0]
    else:
        for i in range(num_bhs):
            masked_subsnap, bhid = next(masked_subsnap_gen)
            bh_id_mask = pygad.IDMask(bhid)
            if snap.bh[bh_id_mask]["mass"] < 1e-15:
                #the BH has 0 mass, most likley due to a merger -> skip this
                if verbose: print(f"Zero-mass BH ({bhid}) detected! Skipping CoM estimate with this BH position as an initial guess")
                continue
            if verbose: print(f"Finding CoM associated with BH ID {bhid}")
            coms[bhid] = pygad.analysis.shrinking_sphere(masked_subsnap, center=snap.bh[bh_id_mask]["pos"], R=initial_radius)
    return coms


def get_com_velocity_of_each_galaxy(snap, xcom, masks=None, min_particle_count=5e4, verbose=True, family="stars"):
    """
    Determine the centre of mass velocity of each galaxy in the simulation,
    assuming each galaxy has an SMBH near its centre.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    xcom : dict
        CoM coordinates for each galaxy, assumes the dict keys are the BH 
        particle ID
    masks : dict, optional
        pygad masks to apply to the (sub) snapshot, by default None
    min_particle_count : float,int, optional
        minimum number of particles to be contained in the sphere, by default 
        5e4
    verbose : bool, optional
        print verbose output, by default True
    family : str, optional
        particle family to analyse, by default "stars"

    Returns
    -------
    vcoms: dict
        keys correspond to the keys in xcom, and the values correspond to the 
        centre of mass velocity of each galaxy
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
        if xcom[idx] is None:
            if verbose:
                print(f"No estimate for CoM associated with BH {idx}. Skipping velocity estimate")
            vcoms[idx] = None
            continue
        #make a ball about the CoM
        ball_radius = np.sort(pygad.utils.dist(masked_subsnap['pos'], xcom[idx]))[int(min_particle_count)]
        if verbose:
            print(f"Maximum radius for velocity CoM set to {ball_radius:.3e} kpc")
        ball_mask = pygad.BallMask(pygad.UnitQty(ball_radius, "kpc"), center=xcom[idx])
        vcoms[idx] = pygad.analysis.mass_weighted_mean(masked_subsnap[ball_mask], qty='vel')
    return vcoms


def get_galaxy_axis_ratios(snap, xcom=None, bin_mask=None, family="stars", return_eigenvectors=False):
    """
    Determine the axis ratios b/a and c/a of a galaxy

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    xcom : pygad.UnitArr, optional
        CoM coordinates for the galaxy, by default None
    bin_mask : pygad.snapshot.masks, optional
        radial or energy masks to apply to the (sub) snapshot, by default None
    family : str, optional
        particle family to analyse, by default "stars"
    return_eigenvectors : bool, optional
        return eigenvectors as well as axis ratios, by default False

    Returns
    -------
    axis_ratios : tuple
        b/a, c/a ratios
    eigenvecs : np.ndarray, optional
        eigenvectors of inertia tensor
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
    if bin_mask is not None:
        #apply either a ball or shell mask
        subsnap = subsnap[bin_mask]
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
    snap : pygad.Snapshot
        _description_
    xcom : dict, optional
        CoM coordinates for each galaxy, assumes the dict keys are the BH 
        particle IDs, by default None
    masks : dict, list, optional
        particle ID masks. Can be either a dict (thus only one particle 
        family goes into the calculation), or a list of dicts (thus stars 
        and dm go into the calculation), by default None

    Returns
    -------
    virial_radius: float, dict
        virial radius of system. If masks were used, this is a dict with the 
        keys being the BH IDs and the mass corresponding to that galaxy as 
        values
    virial_mass: float, dict
        virial mass of system. Same output style as virial_radius

    Raises
    ------
    ValueError
        if mask input is not of correct type
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


def virial_ratio(snap):
    """
    Determine the virial ratio 2K/|W|. Note no centering is done!

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse, must have "pot" block

    Returns
    -------
    : float
        virial ratio
    """
    v2 = pygad.UnitArr(pygad.utils.geo.dist(snap["vel"]))**2
    KK = np.sum(snap["mass"] * v2, axis=-1)
    W = np.sum(snap["mass"] * snap["pot"])
    return np.abs(KK / W)


def calculate_Hamiltonian(snap, chunk=1e5):
    """
    Determine the total Hamiltonian of a system. Requires that ketju has been
    compiled with the:
        OUTPUTPOTENTIAL
    flag (so that the potential is saved to the snapshots)

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    chunk : float,int, optional
        perform summation in chunks of this size for efficiency, by default 1e5

    Returns
    -------
    : float
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
    snap : pygad.Snapshot
        snapshot to analyse

    Returns
    -------
    merged : bool
        has the bh merged?
    remnant_id : int
        ID of the remnant BH (the one with non-zero mass)
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
    bhs : pygad.snapshot.SubSnapshot
        BH subsnapshot, usually of the form snap.bh

    Returns
    -------
    : int
        ID of the more massive of the two BHs
    """
    massive_idx = np.argmax(bhs["mass"])
    return bhs["ID"][massive_idx]


def enclosed_mass_radius(snap, binary=True, mass_frac=1):
    """
    Determine the radius containining the given mass.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    binary : bool, optional
        should the radius be calculated for the binary as a single object 
        (True), or separately for each BH (False)?, by default True
    mass_frac : int, optional
        fraction of the mass to search for. Influence radius corresponds
        to mass_frac = 2., by default 1
    
    Returns
    -------
    r: dict
        keys correspond to BH ID (or the more massive BH ID if binary=True), 
        and values to the radius
    
    Raises
    ------
    AssertionError:
        if snapshot not in physical units
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
    snap : pygad.Snapshot
        snapshot to analyse
    binary : bool, optional
        should the influence radius be calculated for the binary as a single 
        object (True), or separately for each BH (False)?, by default True

    Returns
    -------
    : dict
        keys correspond to BH ID (or the more massive BH ID if binary=True), 
        and values to the influence radius
    """
    return enclosed_mass_radius(snap, binary, mass_frac=2)


def hardening_radius(bhms, rm):
    """
    Determine the hardening radius for the system, defined as Eq. 8.71 in 
    Merritt 2013. This definition uses the influence radius definition of 
    Eq. 2.11.

    Parameters
    ----------
    bhms : list, pygad.UnitArr
        BH masses
    rm : pygad.UnitArr
        influence radius

    Returns
    -------
    : pygad.UnitArr
        hard binary radius in the same units as rm
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
    bh_masses : pygad.UnitArr
        masses of BH binary components
    ah : pygad.UnitArr
        hard binary radius [pc]
    tah : float, pygad.UnitArr
        time of the hard binary radius [Myr]
    H : float
        Quinlan hardening constant
    Gps : float, pygad.UnitArr
        ratio of G * inner density / velocity dispersion [1/(pc * yr)]
    e : int, optional
        initial eccentricity, by default 0

    Returns
    -------
    : pygad.UnitArr
        gravitational wave radius [pc]
    : pygad.UnitArr
        time when the binary is expected to have a = a_GR [Myr]
    
    Raises
    ------
    AssertionError
        if eccentricity is out of interval [0,1]
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
    snap : pygad.Snapshot
        snapshot to analyse
    extent : pygad.UnitArr, optional
        radial range within which to calculate quantities. Default uses all 
        stars in the snapshot, by default None

    Returns
    -------
    inner_density : pygad.UnitArr
        mean stellar mass density within extent
    inner_sigma : pygad.UnitArr
        mean stellar velocity dispersion within extent
    
    Raises
    ------
    AssertionError
        if extent is not a pygad.UnitArr instance
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
    snaplist : list
        snapshots from which to determine the quantity
    t : float
        time to determine the quantity for [Myr]
    extent : pygad.UnitArr, optional
        radial range within which to calculate quantities. Default of None
        uses all stars in the snapshot, by default None

    Returns
    -------
    : pygad.UnitArr
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
    snap : pygad.Snapshot
        snapshot to analyse
    separate_galaxies : bool, optional
        apply a particle ID mask to separate the galaxies, by default True
    shell_kw : dict, optional
        dict to pass np.geomspace controlling the shell properties:
            start: inner radius of inner shell
            stop: outer radius of outer shell
            num: number of shells
            by default {"start":1e-6, "stop":500, "num":20}
    family : str, optional
        particle family to do the analysis for, by default "stars"
    Gcom_kw : dict, optional
        optional arguments to pass to get_com_of_each_galaxy, this
        determines the "global" CoM properties of each galaxy, by default 
        {"initial_radius":10, "min_particle_count":10}
    verbose : bool, optional
        verbose printing?, by default True

    Returns
    -------
    xcoms : dict
        CoM position of each shell
    vcoms : dict
        CoM velocity of each shell
    global_xcom : dict
        global CoM position of each galaxy
    global_vcom : dict
        dict, global CoM velocity of each galaxy
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
        - mass density profile
    of (potentially two) galaxies

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    obs : int, optional
        number of random rotations to perform, by default 10
    family : str, optional
        determine the projected half mass radius for this particle family, by 
        default "stars"
    masks : dict, optional
        dict of id masks (from masks.get_all_id_masks) so that the half mass
        radius can be determined for two galaxies within a merger simulation.
        If None, the system is treated as a whole and the projected half
        mass radius is assigned to just one BH ID., by default None
    q : list, optional
        lower and upper quantiles for error bounds (more robust that std), by 
        default [0.25, 0.75]
    r_edges : array-like, optional
        edges of radial bins for density profile, by default np.geomspace(2e-1, 20, 51)

    Returns
    -------
    Each return variable is a dict with level 1 keys corresponding to the BH
    particle IDs, and level 2 keys corresponding to the lower, estimate, and upper estimates of the value. If no ID masks are given, then only the first
    level 1 key will have data stored with it.
    Re : dict
        effective radius estimates
    vsig : dict
        inner velocity dispersion estimates
    rho : dict
        density profile estimates
    
    Raises
    ------
    AssertionError
        if snapshot not in physical units
    """
    assert(snap.phys_units_requested)
    q.append(0.5)
    q.sort()
    num_bhs = len(snap.bh["mass"])
    if masks is not None:
        assert(len(masks) == num_bhs)
    #pre-allocate dictionaries
    Re = dict.fromkeys(snap.bh["ID"], {"low":0, "estimate":0, "high":0})
    vsig = dict.fromkeys(snap.bh["ID"], {"low":0, "estimate":0, "high":0})
    rho = dict.fromkeys(snap.bh["ID"], {"low":0, "estimate":0, "high":0})
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
            subsnap = getattr(snap, family)
        #centre does not change with rotations
        centre = pygad.analysis.shrinking_sphere(subsnap, centre_guess, 10)
        #temporary arrays to store data
        Re_temp = np.full(obs*3, np.nan)
        vvar_temp = np.full(obs*3, np.nan)
        rho_temp =  np.full((obs*3, len(r_edges)-1), np.nan)
        for i in range(obs):
            rot = pygad.transformation.rot_from_axis_angle(rot_axis[i], rot_angle[i])
            rot.apply(subsnap)
            for proj in range(3):
                linear_idx = 3*i+proj
                Re_temp[linear_idx] = pygad.analysis.half_mass_radius(subsnap, center=centre, proj=proj)
                #we want vel dispersion within Re
                ball_mask = pygad.BallMask(Re_temp[linear_idx], center=centre)
                vvar_temp[linear_idx] = pygad.analysis.los_velocity_dispersion(subsnap[ball_mask], proj=proj)**2
                rho_temp[linear_idx, :] = pygad.analysis.profile_dens(subsnap, qty="mass", r_edges=r_edges, center=centre)
        subsnap.delete_blocks()
        for qi, qkey in zip(q, Re[bhid].keys()):
            Re[bhid][qkey] = pygad.UnitArr(np.nanquantile(Re_temp, qi), units=snap["pos"].units)
            vsig[bhid][qkey] = pygad.UnitArr(np.sqrt(np.nanquantile(vvar_temp, qi)), units=snap["vel"].units)
            rho[bhid][qkey] = pygad.UnitArr(np.nanquantile(rho_temp, qi, axis=0), units=f"({unit_as_str(snap['mass'].units)})/({unit_as_str(snap['pos'].units)}**-2)")
        if masks is None:
            break
    return Re, vsig, rho


def inner_DM_fraction(snap, Re=None, centre=None):
    """
    Determine the dark matter fraction within 1 Re

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    Re : pygad.UnitArr, optional
        effective radius, by default None (calculates the value)
    centre : pygad.UnitArr, optional
        mass centre, default uses the CoM of the BH binary, by default None

    Returns
    -------
    : float
        fraction of DM inside 1 Re
    """
    if Re is None:
        Re,*_ = projected_quantities(snap)
        Re = list(Re.values())[0]["estimate"]
    if centre is None:
        centre = pygad.analysis.center_of_mass(snap.bh)
    ball_mask = pygad.BallMask(Re, center=centre)
    dm_mass = snap.dm["mass"][0]
    star_mass = snap.stars["mass"][0]
    return len(snap.dm[ball_mask])*dm_mass / (len(snap.dm[ball_mask])*dm_mass + len(snap.stars[ball_mask])*star_mass)


def shell_flow_velocities(snap, R, centre=None, direction="out", dt="5 Myr"):
    """
    Return the velocities of those particles moving either inwards or outwards
    through a shell of radius R. This function is largely based on the 
    implementation in pygad.analysis.flow_rates().

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    R : float, pygad.UnitArr
        shell radius, assumed to be in the same units as the position units of 
        snap
    centre : array-like, optional
        centre of shell, by default None
    direction : str, optional
        either "in" or "out", which direction the particles are moving in, by 
        default "out"
    dt : str, optional
        time used for the linear extrapolation of current positions [Myr], by default "5 Myr"

    Returns
    -------
    : np.ndarray
    array of radial velocities corresponding to those particles moving in the 
    desired direction through a shell with radius R.

    Raises
    ------
    AssertionError
        if invalid flow direction given
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


def angular_momentum_difference_gal_BH(snap, mask=None):
    """
    Determine the angular difference between the angular momentum of the stellar
    component of a galaxy and the BHs. Theta is defined as in Nasim et al. 2021
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..498N/abstract 

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    mask : pygad.snapshot.masks, optional
        mask to apply to stellar particles, by default None

    Returns
    -------
    theta : float
        angle between L_gal and L_bh
    L_bh : pygad.UnitArr
        angular momentum of BHs
    L_gal : pygad.UnitArr
        angular momentum of stars (within mask, if specified)
    """
    assert snap.phys_units_requested
    if mask is None:
        L_gal = snap.stars["angmom"].sum(axis=0)
    else:
        L_gal = snap.stars[mask]["angmom"].sum(axis=0)
    L_bh = snap.bh["angmom"].sum(axis=0)
    theta = np.arccos(np.sum(L_gal*L_bh, axis=-1) / (pygad.utils.geo.dist(L_gal) * pygad.utils.geo.dist(L_bh)))
    return theta, L_bh, L_gal


def loss_cone_angular_momentum(snap, a, kappa=1):
    """
    Calculate the approximate angular momentum of the loss cone, as from 
    Gualandris et al. 2017, but multiplied by the stellar mass 
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.2301G/abstract 

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    a : pygad.UnitArr
        BH Binary semimajor axis [pc]
    kappa : float, optional
        dimensionless constant, by default 1

    Returns
    -------
    J : pygad.UnitArr
        Loss cone ang. mom.
    """
    assert snap.phys_units_requested
    J_unit = snap["angmom"].units
    starmass = pygad.UnitScalar(snap.stars["mass"][0], snap.stars["mass"].units)
    Mbin = snap.bh["mass"].sum()
    const_G = const_G = pygad.physics.G.in_units_of("pc/Msol*km**2/s**2")
    J = np.sqrt(2 * const_G * Mbin * kappa * a) * starmass
    return J.in_units_of(J_unit)


def escape_velocity(snap):
    """
    Generate a function that gives the escape velocity as a function of radius

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to construct the fit for

    Returns
    -------
    : function
        interpolation function vesc(r)
    """
    r_pot_interp = scipy.interpolate.interp1d(snap["r"], snap["pot"])
    return lambda r: np.sqrt(2*np.abs(r_pot_interp(r)))


def count_new_hypervelocity_particles(snap, prev=[], vesc=None, family="stars"):
    """
    Determine hypervelocity particles, with consideration of if they have 
    already been counted

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    prev : list, optional
        list of particle IDs that have already been identified as hypervelocity 
        in a previous snapshot, by default []
    vesc : function, optional
        escape velocity function, by default None (calls method to determine)
    family : str, optional
        particle family to analyse, by default "stars"

    Returns
    -------
    : int
        number of new hypervelocity particles
    prev : list
        updated list of hypervelocity IDs
    """
    if vesc is None:
        vesc = escape_velocity(snap)
    subsnap = getattr(snap, family)
    vmag = pygad.utils.geo.dist(subsnap["vel"])
    vesc_eval = vesc(subsnap["r"])
    # determine the particles with a velocity above the escape velocity
    hyper_ids = subsnap["ID"][vmag > vesc_eval]
    # get the IDs of new hypers
    new_hyper_ids = [l for l in hyper_ids if l not in prev]
    prev.extend(new_hyper_ids)
    return len(new_hyper_ids), prev

    