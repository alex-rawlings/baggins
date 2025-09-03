from datetime import datetime
import numpy as np
import scipy.linalg
import scipy.interpolate
import scipy.spatial.transform
import scipy.stats
import pygad
import dask
import baggins.analysis.masks as masks
from baggins.mathematics import (
    radial_separation,
    density_sphere,
    spherical_components,
    get_histogram_bin_centres,
)
from baggins.general import snap_num_for_time, convert_gadget_time, set_seed_time
from baggins.env_config import _cmlogger


__all__ = [
    "basic_snapshot_centring",
    "get_com_of_each_galaxy",
    "get_com_velocity_of_each_galaxy",
    "get_galaxy_axis_ratios",
    "get_virial_info_of_each_galaxy",
    "virial_ratio",
    "calculate_Hamiltonian",
    "determine_if_merged",
    "get_massive_bh_ID",
    "enclosed_mass_radius",
    "lagrangian_radius",
    "influence_radius",
    "hardening_radius",
    "gravitational_radiation_radius",
    "get_inner_rho_and_sigma",
    "get_G_rho_per_sigma",
    "shell_com_motions_each_galaxy",
    "projected_quantities",
    "inner_DM_fraction",
    "shell_flow_velocities",
    "angular_momentum_difference_gal_BH",
    "loss_cone_angular_momentum",
    "escape_velocity",
    "count_new_hypervelocity_particles",
    "velocity_anisotropy",
    "softened_inverse_r",
    "softened_acceleration",
    "add_to_loss_cone_refill",
    "find_individual_bound_particles",
    "relax_time",
    "observable_cluster_props_BH",
    "binding_energy",
    "find_strongly_bound_particles",
]

_logger = _cmlogger.getChild(__name__)


def basic_snapshot_centring(snap):
    """
    Basic-style centring of a snapshot using shrinking sphere method.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to centre (done in place)
    """
    # move to CoM frame
    pre_ball_mask = pygad.BallMask(5)
    centre = pygad.analysis.shrinking_sphere(
        snap.stars,
        pygad.analysis.center_of_mass(snap.stars),
        30,
    )
    vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
    pygad.Translation(-centre).apply(snap, total=True)
    pygad.Boost(-vcom).apply(snap, total=True)


def get_com_of_each_galaxy(
    snap, method="ss", masks=None, family="all", initial_radius=20
):
    """
    Determine the centre of mass of each galaxy in the simulation, assuming each
    galaxy has a single SMBH near its centre.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    method : str, optional
        use minimum potential method (pot) or shrinking sphere method (ss), by
        default "ss"
    masks : dict, optional
        pygad masks to apply to the (sub) snapshot, by default None
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
    assert snap.phys_units_requested
    assert method in ["pot", "ss"]
    num_bhs = len(snap.bh)
    # get IDs corresponding to decreasing mass
    mass_ordered_bh_ids = snap.bh["ID"][np.argsort(-snap.bh["mass"])]

    def _yield_masked_subsnap(s=snap, masks=masks, family=family):
        # helper function to get the maybe masked-, maybe sub-, snapshot
        if masks is None:
            if family == "all":
                for id in mass_ordered_bh_ids:
                    yield (s, id)
            else:
                for id in mass_ordered_bh_ids:
                    yield (getattr(s, family), id)
        else:
            assert len(masks) == num_bhs
            for id in mass_ordered_bh_ids:
                if family == "all":
                    yield (s[masks[id]], id)
                else:
                    ss = getattr(s, family)
                    yield (ss[masks[id]], id)

    coms = dict.fromkeys(snap.bh["ID"], None)
    masked_subsnap_gen = _yield_masked_subsnap()
    if method == "pot":
        for i in range(num_bhs):
            masked_subsnap, bhid = next(masked_subsnap_gen)
            if masks is None:
                if i == 0:
                    min_pot_idx = np.argmin(masked_subsnap["pot"])
                    coms[bhid] = masked_subsnap["pos"][min_pot_idx, :]
                else:
                    coms[bhid] = list(coms.values())[0]
    else:
        zero_mass_flag = False
        for i in range(num_bhs):
            masked_subsnap, bhid = next(masked_subsnap_gen)
            bh_id_mask = pygad.IDMask(bhid)
            if snap.bh[bh_id_mask]["mass"] < 1e-15:
                # the BH has 0 mass, most likley due to a merger -> skip this
                zero_mass_flag = True
                _logger.warning(
                    f"Zero-mass BH ({bhid}) detected! Skipping CoM estimate associated with this BH ID"
                )
                continue
            if masks is None and i > 0 and not zero_mass_flag:
                # we don't want to get two CoMs --> break early
                break
            _logger.debug(f"Finding CoM associated with BH ID {bhid}")
            coms[bhid] = pygad.analysis.shrinking_sphere(
                masked_subsnap,
                center=pygad.analysis.center_of_mass(masked_subsnap),
                R=initial_radius,
            )
    return coms


def get_com_velocity_of_each_galaxy(
    snap, xcom, masks=None, min_particle_count=5e4, family="stars"
):
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
    family : str, optional
        particle family to analyse, by default "stars"

    Returns
    -------
    vcoms: dict
        keys correspond to the keys in xcom, and the values correspond to the
        centre of mass velocity of each galaxy
    """
    assert snap.phys_units_requested
    num_bhs = len(snap.bh["mass"])
    if masks is not None:
        assert len(masks) == num_bhs
    subsnap = getattr(snap, family)
    # prepare the dict that will hold the velocity centre of masses
    vcoms = dict()
    for ind, idx in enumerate(xcom.keys()):
        if masks is not None:
            masked_subsnap = subsnap[masks[idx]]
        else:
            masked_subsnap = subsnap
        if xcom[idx] is None:
            _logger.info(
                f"No estimate for CoM associated with BH {idx}. Skipping velocity estimate"
            )
            vcoms[idx] = None
            continue
        # for very low resolution snaps
        if len(masked_subsnap) < min_particle_count and ind == 0:
            n = len(masked_subsnap)
            while min_particle_count > 0.5 * n:
                min_particle_count *= 0.5
                _logger.warning(
                    f"Particle count is {n}, which is less than the minimum particle count for CoM velocity calculations. Minimum particle count will be set to {min_particle_count}."
                )
        # make a ball about the CoM
        ball_radius = np.sort(pygad.utils.dist(masked_subsnap["pos"], xcom[idx]))[
            int(min_particle_count)
        ]
        _logger.debug(f"Maximum radius for velocity CoM set to {ball_radius:.3e} kpc")
        ball_mask = pygad.BallMask(pygad.UnitQty(ball_radius, "kpc"), center=xcom[idx])
        vcoms[idx] = pygad.analysis.mass_weighted_mean(
            masked_subsnap[ball_mask], qty="vel"
        )
    del subsnap, masked_subsnap
    pygad.gc_full_collect()
    return vcoms


def get_galaxy_axis_ratios(
    snap, bin_mask=None, family="stars", return_eigenvectors=False
):
    """
    Determine the axis ratios b/a and c/a of a galaxy

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
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
    subsnap = getattr(snap, family)
    # move entire subsnap to CoM coordinates
    # just doing this for a masked section results in the radial distance 'r'
    # block not being rederived...
    if bin_mask is not None:
        # apply either a ball or shell mask
        subsnap = subsnap[bin_mask]
    rit = pygad.analysis.reduced_inertia_tensor(subsnap)
    eigen_vals, eigen_vecs = scipy.linalg.eig(rit)
    # need to sort the eigenvalues
    sorted_idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = np.real(eigen_vals[sorted_idx])
    eigen_vecs = eigen_vecs[:, sorted_idx]
    axis_ratios = np.sqrt(eigen_vals[1:] / eigen_vals[0])
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
        snapshot to analyse
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
        return pygad.analysis.virial_info(snap, center=list(xcom.values())[0])
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
        virial_radius[key], virial_mass[key] = pygad.analysis.virial_info(
            snap[masks[key]], center=this_com
        )
    return virial_radius, virial_mass


def virial_ratio(snap):
    """
    Determine the virial ratio abs(2K/W). Note no centering is done!

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse, must have "pot" block

    Returns
    -------
    : float
        virial ratio
    """
    v2 = pygad.UnitArr(pygad.utils.geo.dist(snap["vel"])) ** 2
    KK = np.sum(snap["mass"] * v2, axis=-1)
    W = np.sum(snap["mass"] * snap["pot"])
    return np.abs(KK / W)


def calculate_Hamiltonian(snap, chunk=1e5, return_parts=False):
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
    return_parts : bool
        return KE and PE individually, by default False

    Returns
    -------
    : float
        total energy (the Hamiltonian)
    KE : float, optional
        kinetic energy
    PE : flot, optional
        potential energy
    """
    chunk = int(chunk)
    total_N = snap["pos"].shape[0]
    KE = 0
    PE = 0
    for start in range(0, total_N, chunk):
        end = min(start + chunk, total_N)
        vel_mag = radial_separation(snap["vel"][start:end])
        vel_mag = pygad.UnitArr(vel_mag, "km/s")
        KE += np.sum(0.5 * snap["mass"][start:end] * vel_mag**2)
        PE += np.sum(snap["pot"][start:end] * snap["mass"][start:end])
    if return_parts:
        return KE + PE, KE, PE
    else:
        return KE + PE


def determine_if_merged(snap):
    """
    Determine if a merger has occurred by identifying a BH with 0 mass or
    only one BH present.

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
    # Gadget4 cleans up merged BHs, so there are no ghost particles
    if len(snap.bh) < 2 or np.any(snap.bh["mass"] < 1e-15):
        merged = True
        remnant_id = snap.bh["ID"][snap.bh["mass"] > 1e-15][0]
    else:
        merged = False
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


def _find_radius_for_mass(s, M, centre):
    # determine the radius where the enclosed mass = desired mass M
    r = pygad.utils.geo.dist(s["pos"], centre)
    sorted_idx = np.argsort(r)
    r = r[sorted_idx]
    try:
        assert np.all(np.diff(r) > 0)
    except AssertionError:
        _logger.exception(
            f"Radii are not monotonically ascending! Radii have values {r}",
            exc_info=True,
        )
        raise
    cumul_mass = np.cumsum(s["mass"][sorted_idx])
    try:
        assert np.any(cumul_mass > M)
    except AssertionError:
        _logger.exceptio(
            f"There is not enough mass in stellar particles (max {cumul_mass[-1]:.1e}) to equal desired mass ({M:.1e})",
            exc_info=True,
        )
        raise
    idx = np.argmin(cumul_mass < M)
    r_desired = np.interp(M, cumul_mass[idx - 1 : idx + 1], r[idx - 1 : idx + 1])
    return pygad.UnitScalar(r_desired, units=s["pos"].units, subs=s)


def enclosed_mass_radius(snap, combined=False, mass_frac=1):
    """
    Determine the radius containining a fraction of the mass of the (possibly
    combined) BH.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    combined : bool, optional
        should the radius be calculated for the binary as a single object
        (True), or separately for each BH (False)?, by default False
    mass_frac : float, optional
        fraction of the stellar mass (relative to BH mass) to search for.
        Influence radius corresponds to mass_frac = 2., by default 1.

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
    assert snap.phys_units_requested
    r = dict()
    if combined:
        # we are dealing with the combined mass
        mass_bh = np.sum(snap.bh["mass"])
        centre = pygad.analysis.center_of_mass(snap.bh)
        massive_ID = get_massive_bh_ID(snap.bh)
        _r = _find_radius_for_mass(snap.stars, mass_frac * mass_bh, centre=centre)
        r[massive_ID] = _r
    else:
        # we want the influence radius for each BH. No masking is done to
        # separate the stars to their original galaxy
        bh_idx = np.argsort(snap.bh["mass"])
        for id in snap.bh["ID"][bh_idx]:
            bh_id_mask = pygad.IDMask(id)
            _r = _find_radius_for_mass(
                snap.stars,
                mass_frac * snap.bh[bh_id_mask]["mass"][0],
                centre=snap.bh[bh_id_mask]["pos"].flatten(),
            )
            r[id] = _r
    return r


def lagrangian_radius(snap, mass_frac=0.1, centre=None):
    """
    Determine the Lagrangian radius of a system. Note that a single particle mass is assumed.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    mass_frac : float, optional
        lagrangian radius, by default 0.1
    centre : array-like, optional
        centre position coordinates, by default None

    Returns
    -------
    : pygad.UnitArray
        lagrangian radius from centre
    """
    assert 0 < mass_frac < 1
    target_mass = mass_frac * np.sum(snap.stars["mass"])
    if centre is None:
        centre = pygad.analysis.shrinking_sphere(
            snap.stars,
            center=pygad.analysis.center_of_mass(snap.stars),
            R=np.quantile(snap.stars["r"], 0.75),
        )
    return _find_radius_for_mass(snap.stars, target_mass, centre)


def influence_radius(snap, combined=False):
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
    combined : bool, optional
        should the influence radius be calculated for the binary as a single
        object (True), or separately for each BH (False)?, by default False

    Returns
    -------
    : dict
        keys correspond to BH ID (or the more massive BH ID if combined=True),
        and values to the influence radius
    """
    return enclosed_mass_radius(snap, combined, mass_frac=2)


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
    try:
        assert len(bhms) == 2
    except AssertionError:
        _logger.exception(
            "Hardening radius defined for a BH binary, but only one BH is present!",
            exc_info=True,
        )
        raise
    q = bhms[0] / bhms[1]
    q = 1 / q if q > 1 else q  # ensure q <= 1
    ah = q / (1 + q) ** 2 * rm / 4
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
    # eccentricity function
    Q = (1 - e**2) ** (-3.5) * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    m1_m2_M = np.product(bh_masses) * np.sum(bh_masses)
    # set the constants
    const_G = pygad.physics.G.in_units_of("pc/Msol*km**2/s**2")
    const_c = pygad.physics.c.in_units_of("km/s")
    a_5 = 64 / 5 * const_G**3 * m1_m2_M / (const_c**5 * Gps * H) * Q
    a = pygad.UnitScalar(a_5.in_units_of("pc**5").view(np.ndarray) ** 0.2, "pc")
    time_a = pygad.UnitScalar((ah - a) / (ah * a) / Gps, "Myr") / H + tah
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
        extent_mask = pygad.BallMask(
            extent, center=pygad.analysis.center_of_mass(snap.bh)
        )
        subsnap = snap.stars[extent_mask]
    else:
        _logger.warning("Inner quantities will be calculated for all stars!")
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
        snap = pygad.Snapshot(snaplist[idx + i], physical=True)
        ts[i] = convert_gadget_time(snap, new_unit="Myr")
        rho_temp, sigma_temp = get_inner_rho_and_sigma(snap, extent)
        rho_units = rho_temp.units
        sigma_units = sigma_temp.units
        inner_density[i], inner_sigma[i] = rho_temp, sigma_temp
    if t < ts[0] or t > ts[1]:
        _logger.warning(
            f"Requested time {t} is not within the bounds {ts}. The boundary value in time range will be used."
        )
    f_rho = scipy.interpolate.interp1d(
        ts, inner_density, bounds_error=False, fill_value=tuple(inner_density)
    )
    f_sigma = scipy.interpolate.interp1d(
        ts, inner_sigma, bounds_error=False, fill_value=tuple(inner_sigma)
    )
    G_rho_per_sigma = (
        pygad.physics.G
        * pygad.UnitScalar(f_rho(t), rho_units)
        / pygad.UnitScalar(f_sigma(t), sigma_units)
    )
    return G_rho_per_sigma.in_units_of("pc**-1/yr")


def shell_com_motions_each_galaxy(
    snap,
    separate_galaxies=True,
    shell_kw={"start": 1e-6, "stop": 500, "num": 20},
    family="stars",
    Gcom_kw={"initial_radius": 10, "min_particle_count": 10},
    verbose=True,
):
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
        # mask the particles as belonging to one of two progenitors
        id_masks = masks.get_all_id_masks(snap, family=family)
    else:
        subsnap = getattr(snap, family)
        id_masks = {snap.bh["ID"][0]: pygad.IDMask(subsnap["ID"])}
    xcoms = dict()
    vcoms = dict()
    for k in id_masks.keys():
        xcoms[k] = np.full((shell_kw["num"], 3), np.nan)
        vcoms[k] = np.full((shell_kw["num"], 3), np.nan)
    global_xcom = get_com_of_each_galaxy(
        snap, family=family, verbose=verbose, masks=id_masks, **Gcom_kw
    )
    global_vcom = get_com_velocity_of_each_galaxy(
        snap, global_xcom, masks=id_masks, family=family, verbose=verbose
    )
    shell_radii = np.geomspace(**shell_kw)
    # iterate over each shell
    for i, (r_inner, r_outer) in enumerate(zip(shell_radii[:-1], shell_radii[1:])):
        # mask particles to this shell
        radial_mask = masks.get_all_radial_masks(
            snap,
            (r_inner, r_outer),
            centre=global_xcom,
            id_masks=id_masks,
            family=family,
        )
        # compute CoM motions for shell using mass-weighted means
        for bhid in id_masks.keys():
            xcoms[bhid][i, :] = pygad.analysis.mass_weighted_mean(
                snap[radial_mask[bhid]], qty="pos"
            )
            vcoms[bhid][i, :] = pygad.analysis.mass_weighted_mean(
                snap[radial_mask[bhid]], qty="vel"
            )
    return xcoms, vcoms, global_xcom, global_vcom


def projected_quantities(
    snap,
    obs=10,
    family="stars",
    masks=None,
    r_edges=np.geomspace(2e-1, 20, 51),
    rng=None,
):
    """
    Determine projected quantities of:
    - half mass radius,
    - velocity dispersion profile
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
    r_edges : array-like, optional
        edges of radial bins for density profile, by default np.geomspace(2e-1, 20, 51)
    rng : np.random._generator.Generator, optional
        random number generator, by default None (creates a new instance)

    Returns
    -------
    eff_rad : dict
        effective radius samples for each galaxy (if no ID masks are given, then only one key exists.)
    vsig2_re : dict
        square of velocity dispersion within 1Re, as per eff_rad
    vsig2_r : dict
        square of velocity dispersion as a function of r, as per eff_rad
    surf_rho : dict
        density profile estimate, format as per eff_rad

    Raises
    ------
    AssertionError
        if snapshot not in physical units
    ValueError
        if masking attempted without BHs in snapshot
    """
    assert snap.phys_units_requested
    num_bhs = len(snap.bh["mass"])
    if masks is not None:
        assert len(masks) == num_bhs

    # shrinking sphere scaling
    shrink_sphere_r0 = {"stars": 10.0, "dm": 200.0}
    try:
        shrink_sphere_r0[family]
    except KeyError:
        _logger.warning(
            f"Shrinking sphere initial guess not defined for '{family}', using value for 'stars' instead."
        )
        shrink_sphere_r0[family] = shrink_sphere_r0["stars"]

    # set up rng and distributions
    if rng is None:
        rng = np.random.default_rng(set_seed_time())
    rot_axis = rng.uniform(-1, 1, (obs, 3))
    rot_angle = rng.uniform(0, np.pi, obs)
    now = datetime.now()

    def _helper_func(s, proj, re, c):
        """
        Helper function to pass to parallel methods for projected quantity
        calculation

        Parameters
        ----------
        s : pygad.Snapshot
            snapshot to analyse
        proj : int
            axis to project along
        re : array-like
            radial bin edges
        c : array-like
            coordinates of centre

        Returns
        -------
        _eff_rad : pygad.UnitArr
            effective radius for this projection
        _vsig2_re : pygad.UnitArr
            square velocity dispersion within 1Re
        _vsig2_r : pygad.UnitArr
            square of velocity dispersion as a function of r
        _surf_rho : pygad.UnitArr
            projected mass surface density profile
        """
        _eff_rad = pygad.analysis.half_mass_radius(s, center=c, proj=proj)
        # vel dispersion within Re
        ball_mask = pygad.BallMask(_eff_rad, center=c)
        _vsig2_re = pygad.analysis.los_velocity_dispersion(s[ball_mask], proj=0) ** 2
        rbin_dict = pygad.analysis.get_radial_bins(s=s, r_edges=re, proj=proj, center=c)
        vel_projs = ["vx", "vy", "vz"]
        _vsig2_r = pygad.analysis.radially_binned_statistic(
            s=s,
            qty=f"{vel_projs[proj]}",
            proj=proj,
            center=c,
            rdict=rbin_dict,
            statistic=np.nanvar,
        )
        _surf_rho = pygad.analysis.profile_dens(
            s, qty="mass", center=c, proj=proj, rdict=rbin_dict
        )
        return _eff_rad, _vsig2_re, _vsig2_r, _surf_rho

    # determine if we are masking or not
    centre_guess_dict = {}
    subsnap_dict = {}
    if masks is None:
        if num_bhs > 0:
            bhid = snap.bh["ID"][0]
        else:
            _logger.info(f"Number of BHs present: {num_bhs}")
            bhid = 0
        centre_guess_dict[bhid] = pygad.analysis.center_of_mass(snap.stars)
        subsnap_dict[bhid] = getattr(snap, family)
    else:
        if num_bhs < 1:
            raise ValueError("BHs must be present in snapshot for masking!")
        for bhid in snap.bh["ID"]:
            centre_guess_dict[bhid] = snap.stars[snap.bh["ID"] == bhid]["pos"]
            subsnap_dict[bhid] = getattr(snap[masks[bhid]], family)

    # pre-allocate dictionaries
    eff_rad = dict.fromkeys(centre_guess_dict.keys(), np.full(3 * obs, np.nan))
    vsig2_re = dict.fromkeys(centre_guess_dict.keys(), np.full(3 * obs, np.nan))
    vsig2_r = dict.fromkeys(
        centre_guess_dict.keys(), np.full((3 * obs, len(r_edges) - 1), np.nan)
    )
    surf_rho = dict.fromkeys(
        centre_guess_dict.keys(), np.full((3 * obs, len(r_edges) - 1), np.nan)
    )

    # for each distinct galaxy in the sim
    for j, bhid in enumerate(centre_guess_dict.keys()):
        centre = pygad.analysis.shrinking_sphere(
            subsnap_dict[bhid], centre_guess_dict[bhid], shrink_sphere_r0[family]
        )
        _logger.debug(
            f"Difference between centre guess and shrinking sphere centre is: {centre_guess_dict[bhid][0] - centre}"
        )
        for o in range(obs):
            # rotate the snapshot, and independently the CoM
            rot = pygad.transformation.rot_from_axis_angle(
                rot_axis[o] - centre, rot_angle[o]
            )
            rot.apply(subsnap_dict[bhid], total=True)
            _rot = scipy.spatial.transform.Rotation.from_matrix(rot.rotmat)
            centre = _rot.apply(centre)
            results = []
            # set up the parallelism
            for job in range(3):
                results.append(
                    dask.delayed(_helper_func)(subsnap_dict[bhid], job, r_edges, centre)
                )
            results = dask.compute(*results)
            # move results to a better format
            for i, r in enumerate(results):
                idx_0 = 3 * o + i
                idx_1 = 3 * (o + 1) + i
                eff_rad[bhid][idx_0] = r[0]
                vsig2_re[bhid][idx_0] = r[1]
                vsig2_r[bhid][idx_0:idx_1] = r[2]
                surf_rho[bhid][idx_0:idx_1] = r[3]
        subsnap_dict[bhid].delete_blocks()
        subsnap_dict[bhid] = None
    _logger.info(f"Projected quantities determined in {datetime.now()-now}")
    return eff_rad, vsig2_re, vsig2_r, surf_rho


def inner_DM_fraction(snap, Re=None):
    """
    Determine the dark matter fraction within 1 Re

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    Re : pygad.UnitArr, optional
        effective radius, by default None (calculates the value)

    Returns
    -------
    : float
        fraction of DM inside 1 Re
    """
    if Re is None:
        Re, *_ = projected_quantities(snap)
        Re = np.nanmedian(list(Re.values())[0])
    ball_mask = pygad.BallMask(Re)
    dm_mass = snap.dm["mass"][0]
    star_mass = snap.stars["mass"][0]
    return (
        len(snap.dm[ball_mask])
        * dm_mass
        / (len(snap.dm[ball_mask]) * dm_mass + len(snap.stars[ball_mask]) * star_mass)
    )


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
    # this step is to prevent the conversion of an entire array
    dt.convert_to(snap["r"].units / snap["vel"].units)
    if centre is None:
        centre = [0.0, 0.0, 0.0]
    centre = pygad.UnitArr(centre, units=snap["r"].units)
    t = pygad.Translation(-centre)
    # radial velocity appears to be updated after translation too
    t.apply(snap)
    rpred = snap["r"] + snap["vrad"] * dt
    if direction == "out":
        # particles are less than R originally, but move outwards
        mask = (snap["r"] < R) & (rpred >= R)
    else:
        # particles are further than R originally, but move inwards
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
    theta = np.arccos(
        np.sum(L_gal * L_bh, axis=-1)
        / (pygad.utils.geo.dist(L_gal) * pygad.utils.geo.dist(L_bh))
    )
    return theta, L_bh, L_gal


def loss_cone_angular_momentum(snap, a, e=0, kappa=None):
    """
    Calculate the approximate angular momentum of the loss cone, as per
    Bortolas 2016 or Gualandris 2017, but multiplied by the stellar mass
    https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.1023B/abstract
    or
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.2301G/abstract
    Note the first definition follows from the definition of the semimajor axis
    in Binney and Tremaine. The second definition requires a scaling parameter
    kappa, typically taken to be 1.


    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    a : pygad.UnitArr
        BH Binary semimajor axis [pc]
    e : float, optional
        BH Binary eccentricity
    kappa : float, optional
        dimensionless constant used to scale semimajor axis in Gualandris
        implementation, by default None (Bortolas implementation used)

    Returns
    -------
    J : pygad.UnitArr
        loss cone ang. mom.
    """
    assert snap.phys_units_requested
    J_unit = snap["angmom"].units
    starmass = pygad.UnitScalar(snap.stars["mass"][0], snap.stars["mass"].units)
    Mbin = snap.bh["mass"].sum()
    const_G = pygad.physics.G.in_units_of("pc/Msol*km**2/s**2")
    if kappa is None:
        J = np.sqrt(const_G * Mbin * a * (1 - e**2)) * starmass
    else:
        _logger.debug(
            "Loss cone angular momentum determined without accounting for eccentricity."
        )
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
    idx = np.argsort(snap["r"])
    return lambda r: np.sqrt(2 * np.abs(np.interp(r, snap["r"][idx], snap["pot"][idx])))


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
    new_hyper_ids = [n for n in hyper_ids if n not in prev]
    prev.extend(new_hyper_ids)
    return len(new_hyper_ids), prev


def velocity_anisotropy(
    snap, r_edges, xcom=[0, 0, 0], vcom=[0, 0, 0], qcut=1.0, eps=1e-16
):
    """
    Determine the beta profile for a snapshot.

    Parameters
    ----------
    snap : pygad.Snapshot
        snap to analyse
    r_edges : array-like
        edges of radial bins for beta profile
    xcom : list or array-like, optional
        positional centre of mass, by default [0,0,0]
    vcom : list or array-like, optional
        velocity centre of mass, by default [0,0,0]
    qcut : float, optional
        filter out particles above qcut quantile in velocity magnitude, by
        default 1.0
    eps : float, optional
        tolerance to prevent zero-division, by default 1e-16

    Returns
    -------
    np.ndarray
        beta profile as a function of radius
    np.ndarray
        number of particles per radial bin
    """
    if isinstance(xcom, list):
        xcom = np.array(xcom)
    if isinstance(vcom, list):
        vcom = np.array(vcom)
    pygad.Translation(-xcom).apply(snap)
    pygad.Boost(-vcom).apply(snap)
    r = pygad.utils.geo.dist(snap["pos"])
    v_sphere = spherical_components(snap["pos"], snap["vel"])
    if qcut < 1:
        vmag = radial_separation(snap["vel"])
        mask = vmag < np.nanquantile(vmag, qcut)
        r = r[mask]
        v_sphere = v_sphere[mask, :]
    # bin statistics
    standard_devs, *_ = scipy.stats.binned_statistic(
        r, [v_sphere[:, i] for i in range(3)], statistic="std", bins=r_edges
    )
    bin_counts, *_ = np.histogram(r, bins=r_edges)
    # determine beta(r)
    beta = 1 - (standard_devs[1, :] ** 2 + standard_devs[2, :] ** 2) / (
        2 * standard_devs[0, :] ** 2 + eps
    )
    return beta, bin_counts


def softened_inverse_r(r, h):
    """
    Return the softened value of 1/r following the Gadget kernel

    Parameters
    ----------
    r : array-like
        radial distances of particles
    h : float
        softening value

    Returns
    -------
    _inv_r_soft : array-like
        softened inverse r array
    """
    if h is None or h <= 0:
        return 1 / r
    hinv = 1 / h
    u = r.view(np.ndarray) * hinv
    _inv_r_soft = np.full(r.shape, np.nan)

    # mask the different sections
    mask = u >= 1
    _inv_r_soft[mask] = 1 / r[mask]

    mask = u < 0.5
    _inv_r_soft[mask] = -hinv * (
        -2.8 + u[mask] ** 2 * (16 / 3 + u[mask] ** 2 * (6.4 * u[mask] - 9.6))
    )

    mask = np.logical_not(np.logical_and(u >= 1, u < 0.5))
    _inv_r_soft[mask] = -hinv * (
        -3.2
        + 2 / 30 / u[mask]
        + u[mask] ** 2
        * (32 / 3 + u[mask] * (-16 + u[mask] * (9.6 - 32 / 15 * u[mask])))
    )
    return _inv_r_soft


def softened_acceleration(
    snap, h={"stars": None, "dm": None, "bh": None}, centre=[0, 0, 0], exclude_id=[]
):
    """
    Determine the Gadget-softened acceleration

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    h : dict, optional
        particle family softenings, by default {"stars":None, "dm":None, "bh":None}
    centre : array-like, optional
        centre to determine r from, by default [0,0,0]
    exclude_id : list, optional
        particle IDs to exclude from the calculation, by default []

    Returns
    -------
    accel : array-like
        vector of accelerations evaluated at position `centre`
    """
    accel = np.zeros(3)
    if exclude_id:
        id_mask = pygad.IDMask(exclude_id)
    else:
        id_mask = None
    for family in ["stars", "dm", "bh"]:
        _logger.debug(f"Determining acceleration for family {family}")
        subsnap = getattr(snap, family)
        if id_mask is not None:
            subsnap = subsnap[~id_mask]
        r = pygad.utils.geo.dist(subsnap["pos"], centre)
        inv_r = softened_inverse_r(r, h[family])
        accel += np.sum(
            np.atleast_2d(subsnap["mass"] * inv_r**3).T * subsnap["pos"], axis=0
        )
    return accel


def add_to_loss_cone_refill(snap, J_lc, prev):
    """
    Determine the new particles in the binary loss cone. Note this method is designed to be called within a construct that loops over snapshots.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse, potentially a masked subsnapshot
    J_lc : float, pygad.UnitQty
        loss cone angular momentum of binary
    prev : set
        particles in loss cone in previous snapshots

    Returns
    -------
    : set
        updated set of particles IDs that have been in binary loss cone
    """
    in_cone_ids = set(snap["ID"][pygad.utils.geo.dist(snap["angmom"]) < J_lc])
    return prev.union(in_cone_ids)


def _set_bound_search_rad(snap):
    """
    Define the search area for bound stellar particles. The search is
    restricted to the influence radius of the most massive BH, centred on that
    BH.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse

    Returns
    -------
    pygad.SubSnapshot
        subsnapshot with just those particles to search for boundedness
    """
    rinf = influence_radius(snap)
    bh_id = get_massive_bh_ID(snap)
    ball_mask = pygad.BallMask(rinf[bh_id], snap.bh[snap.bh["ID"] == bh_id]["pos"])
    return snap[pygad.IDMask(snap.bh["ID"]) | pygad.IDMask(snap.stars["ID"])][ball_mask]


def find_individual_bound_particles(snap, return_extra=False):
    """
    Find individual particles bound to the most massive BH (two-body energy is
    checked).

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    return_extra : bool, optional
        return extra information, including:
        - the fraction of bound particles inside the influence radius,
        - BH energy
        by default False

    Returns
    -------
    : list
        list of bound particle IDs
    : float, optional
        fraction of bound particles inside influence radius if `return_extra` is
        True
    : array-like, optional
        particle energies of bound particles if `return_extra` is True
    """
    subsnap = _set_bound_search_rad(snap)
    bh_id_mask = pygad.IDMask(get_massive_bh_ID(subsnap.bh))
    # shift to BH frame
    # note we have to shift the whole snapshot for the transformation to work
    trans = pygad.Translation(-subsnap[bh_id_mask]["pos"][0, :])
    boost = pygad.Boost(-subsnap.bh[bh_id_mask]["vel"][0, :])
    trans.apply(subsnap, total=True)
    boost.apply(subsnap, total=True)
    try:
        assert np.all(subsnap.bh[bh_id_mask]["pos"] < 1e-12)
        assert np.all(subsnap.bh[bh_id_mask]["vel"] < 1e-12)
    except AssertionError:
        _logger.exception(
            f"The centering has not worked, cannot find the bound particles! The BH has position {subsnap.bh[bh_id_mask]['pos']} and velocity {subsnap.bh[bh_id_mask]['vel']}",
            exc_info=True,
        )
        raise
    G = pygad.UnitScalar(4.3009e-6, "kpc/Msol*(km/s)**2")
    KE = pygad.UnitArr(
        0.5 * np.linalg.norm(subsnap[~bh_id_mask]["vel"], axis=1) ** 2, "(km/s)**2"
    )
    PE = (
        G
        * pygad.UnitScalar(subsnap.bh[bh_id_mask]["mass"][0], "Msol")
        / pygad.UnitArr(subsnap[~bh_id_mask]["r"], snap["r"].units)
    )
    bound_IDs = subsnap[~bh_id_mask][KE - PE < 0]["ID"]
    # now we put the snapshot back in the original coordinate system
    # ie we apply the inverse transformations
    trans.inverse().apply(subsnap, total=True)
    boost.inverse().apply(subsnap, total=True)
    if return_extra:
        return bound_IDs, len(bound_IDs) / len(subsnap), KE - PE
    else:
        return bound_IDs


def relax_time(snap, r):
    """
    Half mass relaxation time, as given in Binney and Tremaine 2008 Eq. 7.107

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    r : float, pygad.UnitArr
        radius to determine relaxation time for

    Returns
    -------
    tr : pygad.UnitArr
        relaxation time
    """
    try:
        assert len(np.unique(snap.stars["mass"])) == 1
        star_mass = np.mean(
            [
                pygad.UnitScalar(snap.stars["mass"][0], "Msol"),
                pygad.UnitScalar(snap.dm["mass"][0], "Msol"),
            ]
        )
        star_mass = pygad.UnitScalar(star_mass, snap["mass"].units)
    except AssertionError:
        _logger.exception(
            "Calculation only valid for stellar particles with constant mass!",
            exc_info=True,
        )
    if not isinstance(r, pygad.UnitArr):
        r = pygad.UnitScalar(r, snap["pos"].units)
    id_mask = pygad.IDMask(get_massive_bh_ID(snap))
    centre = pygad.analysis.shrinking_sphere(snap.stars, snap.bh[id_mask]["pos"], 30)
    ball_mask = pygad.BallMask(r, centre)
    G = pygad.physics.G.in_units_of("kpc/Msol*(km/s)**2")
    coulomb = np.log(
        r
        * np.mean(radial_separation(snap[ball_mask]["vel"]) ** 2)
        / (2 * G * star_mass)
    )
    tr = (
        2.1
        * pygad.UnitScalar(np.std(snap[ball_mask]["vel"]), "km/s")
        * r**2
        / (G * star_mass * coulomb)
    )
    return tr.in_units_of("Gyr")


def observable_cluster_props_BH(
    snap, Rmax=30, n_gal_bins=51, n_cluster_bins=21, proj=None, vel_clip=None
):
    """
    Find the density of an cluster around a SMBH, comparing it to the background host galaxy density. Note NO centring is done. If 'proj' is None, then a 3D density profile is produced, otherwise a 2D density.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    Rmax : float, optional
        maximum galaxy radius, by default 30
    n_gal_bins : int, optional
        number of bins for galaxy density profile, by default 51
    n_cluster_bins : int, optional
        number of bins for cluster density profile, by default 21
    proj : int, optional
        projection, by default None

    Returns
    -------
    obs_props : dict
        observationaal properties of cluster
    """
    try:
        assert len(snap.bh) == 1
    except AssertionError:
        _logger.exception(
            f"Only 1 BH can be present, there are {len(snap.bh)}", exc_info=True
        )
        raise
    bound_mask = pygad.IDMask(find_individual_bound_particles(snap))

    # helper function to get the density of the total cluster
    def _get_cluster_density(_Rmax):
        r_edges = np.geomspace(1e-2, _Rmax, n_cluster_bins)
        r_centres = get_histogram_bin_centres(r_edges)
        dens = pygad.analysis.profile_dens(
            snap.stars[bound_mask],
            "mass",
            r_edges=r_edges,
            center=snap.bh["pos"].flatten(),
            proj=proj,
        )
        return r_centres + snap.bh["r"].flatten(), dens

    # get the density of the entire galaxy
    r_edges_gal = np.geomspace(1e-2, Rmax, n_gal_bins)
    r_centres_gal = get_histogram_bin_centres(r_edges_gal)
    gal_dens = pygad.analysis.profile_dens(
        snap.stars, "mass", r_edges=r_edges_gal, proj=proj
    )
    # remove NaNs from galaxy density
    nan_idx = np.isnan(gal_dens)
    r_centres_gal = r_centres_gal[~nan_idx]
    gal_dens = gal_dens[~nan_idx]
    try:
        np.all(np.diff(r_centres_gal) > 0)
    except AssertionError:
        _logger.exception("Radii not monotonically increasing", exc_info=True)
        raise

    # only include those stars which are within a radius that encloses a
    # density above the background density
    rinfl = list(influence_radius(snap).values())[0]
    r_centres_cluster, cluster_dens = _get_cluster_density(rinfl)
    background_dens = np.interp(r_centres_cluster, r_centres_gal, gal_dens, left=0)
    if np.all(cluster_dens < background_dens):
        visible = False
    elif np.all(cluster_dens > background_dens):
        _logger.debug("All of cluster is visible above background")
        visible = True
        cluster_max_r = rinfl
    else:
        cluster_max_r_idx = (
            np.argmin(cluster_dens > background_dens) + 1
        )  # plus 1 so we get the full bin
        cluster_max_r = r_centres_cluster[cluster_max_r_idx] - snap.bh["r"].flatten()
        # now fit the cluster that is visible above the background
        r_centres_cluster, cluster_dens = _get_cluster_density(cluster_max_r[0])
        if r_centres_cluster[0] < r_centres_gal[0]:
            # protect against low-particle noise in centre
            visible = False
        else:
            visible = np.any(cluster_dens > background_dens)
    obs_props = dict(cluster_mass=None, cluster_Re_pc=None, cluster_vsig=None)
    if visible:
        if proj is None:
            # we are taking a 3D Ball within which we calculate cluster props
            cluster_mask = pygad.BallMask(
                cluster_max_r[0], center=snap.bh["pos"].flatten()
            )
        else:
            # we are taking a cylinder with a projected radius the same as the
            # cluster
            xyaxis = list(set({0, 1, 2}).difference({proj}))
            cluster_mask = masks.get_cylindrical_mask(
                cluster_max_r[0], proj=proj, centre=snap.bh["pos"][:, xyaxis].flatten()
            )
        obs_props["cluster_mass"] = np.sum(snap.stars[cluster_mask]["mass"])
        obs_props["cluster_Re_pc"] = pygad.analysis.half_mass_radius(
            snap.stars[cluster_mask], center=snap.bh["pos"].flatten(), proj=proj
        ).in_units_of("pc", subs=snap)
        if proj is None:
            # use a 3D half mass radius
            cluster_re_mask = pygad.BallMask(
                obs_props["cluster_Re_pc"], center=snap.bh["pos"].flatten()
            )
        else:
            cluster_re_mask = masks.get_cylindrical_mask(
                obs_props["cluster_Re_pc"],
                proj=proj,
                centre=snap.bh["pos"][:, xyaxis].flatten(),
            )
        cluster_vel = snap.stars[cluster_re_mask]["vel"]
        if proj is not None:
            cluster_vel = cluster_vel[:, proj]
        cluster_vel = cluster_vel.flatten()
        if vel_clip is not None:
            # apply an optional sigma clipping to protect against outliers
            vsig = np.nanstd(cluster_vel)
            cluster_vel = cluster_vel[np.abs(cluster_vel) < vel_clip * vsig]
        obs_props["cluster_vsig"] = np.nanstd(cluster_vel)
    obs_props.update(
        dict(
            r_centres_cluster=r_centres_cluster,
            cluster_dens=cluster_dens,
            r_centres_gal=r_centres_gal,
            gal_dens=gal_dens,
            visible=visible,
        )
    )
    return obs_props


def binding_energy(snap):
    """
    Calculate the binding energy of each particle. Note that no centring is
    done.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse

    Returns
    -------
    : array
        binding energy for each particle in snap
    """
    return (
        -snap["mass"] * snap["pot"]
        - 0.5 * snap["mass"] * pygad.utils.geo.dist(snap["vel"]) ** 2
    )


def find_strongly_bound_particles(snap, rfac=5, return_extra=False):
    """
    Find those particles within the influence radius of the most massive BH
    which are strongly bound to it, compared to the ambient velocity dispersion.
    This method calls find_individual_bound_particles() to get the initial set
    of bound particles.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to analyse
    rfac : float, optional
        ambient vel. dispersion determined within this many influence radii, by default 5
    return_extra : bool, optional
        return extra information, including:
        - energies of particles within influence radius
        - ambient velocity dispersion
        by default False

    Returns
    -------
    : list
        list of strongly bound particle IDs
    energy : array-like, optional
        energies if 'return_extra' is True
    ambient_sigma : float, optional
        ambient velocity dispersion if 'return_extra' is True
    """
    bound_ids, _, energy = find_individual_bound_particles(snap, return_extra=True)
    # we want those particles which are strongly bound
    ambient_ball = pygad.BallMask(
        rfac * list(influence_radius(snap).values())[0], center=snap.bh["pos"].flatten()
    )
    ambient_sigma = np.linalg.norm(np.std(snap.stars[ambient_ball]["vel"], axis=0))
    bound_id_mask = pygad.IDMask(bound_ids[energy[energy < 0] / ambient_sigma**2 < -1])
    if return_extra:
        return snap.stars[bound_id_mask]["ID"], energy, ambient_sigma
    else:
        return snap.stars[bound_id_mask]["ID"]
