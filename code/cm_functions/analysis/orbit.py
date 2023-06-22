import numpy as np
import scipy.spatial.distance, scipy.signal, scipy.optimize, scipy.integrate, scipy.interpolate
import ketjugw
from ..general import get_idx_in_array
from ..mathematics import radial_separation, angle_between_vectors, project_orthogonal
from ..env_config import _cmlogger

__all__ = ["find_pericentre_time", "interpolate_particle_data", "get_bh_particles", "get_bound_binary", "get_binary_before_bound", "linear_fit_get_H", "linear_fit_get_K", "analytic_evolve_peters_quinlan", "get_hard_timespan", "find_idxs_of_n_periods", "impact_parameter", "move_to_centre_of_mass", "deflection_angle", "first_major_deflection_angle"]

_logger = _cmlogger.copy(__file__)

#common units
myr = 1e6 * ketjugw.units.yr
kpc = 1e3 * ketjugw.units.pc


def find_pericentre_time(bh1, bh2, height=-10, return_sep=False, **kwargs):
    """
    Determine the time (in Myr) of pericentre passages of the BHs with the 
    scipy.signals.find_peaks() function

    Parameters
    ----------
    bh1 : ketjugw.Particle
        ketju bh
    bh2 : ketjugw.Particle
        ketju bh
    height : float, optional
        peaks required to have a height greater than this, by default -10
    return_sep : bool, optional
        return the positional-separation array, by default False
    kwargs: other keyword arguments for scipy.signal.find_peaks()

    Returns
    -------
    : float
        time of pericentre passages
    peak_idxs : np.ndarray
        index position of pericentre passage in BH Particle attribute arrays
    sep : np.ndarray, optional
        separation between the BHs as a function of time
    """
    sep = radial_separation(bh1.x/kpc, bh2.x/kpc)
    #pericentre is found by negating the separation, and identifying those 
    #peaks close to 0
    peak_idxs = scipy.signal.find_peaks(-sep, height=height, **kwargs)[0]
    if return_sep:
        return bh1.t[peak_idxs], peak_idxs, sep*kpc
    else:
        return bh1.t[peak_idxs], peak_idxs


def interpolate_particle_data(p_old, t):
    """
    Interpolate the attributes of particle bhold to the times specifed by t.

    Parameters
    ----------
    p_old : ketjugw.Particle
        object to do the interpolation for
    t : np.ndarray
        times to interpolate data to

    Returns
    -------
    p_new : ketjugw.Particle
        object with interpolated data
    """
    #initialise the new particle object
    p_new = ketjugw.Particle(-99, 0, [0,0,0], [0,0,0])
    setattr(p_new, "t", t)
    for a in ("m", "x", "v", "spin"):
        v = getattr(p_old, a)
        finterp = scipy.interpolate.interp1d(p_old.t, v, axis=0)
        setattr(p_new, a, finterp(t))
    return p_new


def get_bh_particles(ketju_file, tol=1e-15):
    """
    Return the bh particles in the (usually-named) ketju_bhs.hdf5 file.
    This is really just a wrapper that ensures:
        - the BHs are still separate particles (if a merger occurs, the time
          series of one particle will be longer than the other)
        - the BHs have the same time domain (Gadget-only integration will 
          generally produce bh1.t[i] != bh2.t[i], causing problems for later
          calculations). Interpolation is performed to overcome this, if
          necessary.

    Parameters
    ----------
    ketju_file : str
        path to ketju_bhs.hdf5 file to analyse
    tol : float, optional
        tolerance for equality testing, by default 1e-15

    Returns
    -------
    bh1 : ketjugw.Particle
        object for the BH (will be named bh1interp if interpolation was 
        performed)
    bh2 : ketjugw.Particle
        same as bh1, but for bh2
    merged : MergerInfo
        class containing merger remnant info
    """
    bh1, bh2 = ketjugw.data_input.load_hdf5(ketju_file).values()
    len1, len2 = len(bh1), len(bh2)
    min_len = min(len1, len2)
    merged = MergerInfo()
    # first need to determine if time series are consistent between particles
    if np.any(np.abs(bh1.t[:min_len] - bh2.t[:min_len])>tol):
        # particle time series are not in sync, need to interpolate
        # merger only occurs if Ketju is activated, in which case the time
        # series are in sync by construction, so no merger has occurred here
        # TODO is there a more robust way to ascertain if a merger has (not)
        # occurred that doesn't tie us to how Ketju data output occurs
        _logger.logger.warning("Particle time series are not consistent with each other: linear interpolation will be performed")
        t_arr = np.linspace(max(bh1.t[0], bh2.t[0]), min(bh1.t[-1], bh2.t[-1]), max(len1, len2))
        bh1interp = interpolate_particle_data(bh1, t_arr)
        bh2interp = interpolate_particle_data(bh2, t_arr)
        return bh1interp, bh2interp, merged
    else:
        # particles are in sync, and we can return as normal
        if len1 > len2:
            #bh2 has merged into bh1
            merged.update(bh1[len2])
            bh1 = bh1[:len2]
        elif len1 < len2:
            #bh1 has merged into bh2
            merged.update(bh2[len1])
            bh2 = bh2[:len1]
        return bh1, bh2, merged


def get_bound_binary(ketju_file, tol=1e-15):
    """
    Return the data from the ketju_bhs.hdf5 file corresponding to when the 
    binary becomes (and remains) bound. 

    Parameters
    ----------
    ketju_file : str
        path to ketju_bhs.hdf5 file to analyse
    tol : float, optional
        tolerance for equality testing, by default 1e-15

    Returns
    -------
    bh1 : ketjugw.Particle
        object for bh1, where the particle has the same time domain as bh2
    bh2 : ketjugw.Particle
        same as bh1, but for bh2
    merged: MergerInfo
        class containing merger remnant info
    """
    bh1, bh2, merged = get_bh_particles(ketju_file, tol)
    bhs = {0:bh1, 1:bh2}
    try:
        bh1, bh2 = list(ketjugw.find_binaries(bhs, remove_unbound_gaps=True).values())[0]
    except IndexError:
        _logger.logger.exception("No binaries found!", exc_info=True)
        raise
    return bh1, bh2, merged


def get_binary_before_bound(ketju_file, tol=1e-15):
    """
    Return the data from the ketju_bhs.hdf5 file corresponding to before when 
    the binary becomes (and remains) bound. 

    Parameters
    ----------
    ketju_file : str
        path to ketju_bhs.hdf5 file to analyse
    tol : float, optional
        tolerance for equality testing, by default 1e-15

    Returns
    -------
    bh1 : ketjugw.Particle
        object for bh1, where the particle has the same time domain as bh2
    bh2 : ketjugw.Particle
        same as bh1, but for bh2
    bound_state : str
        how binary is behaving
    """
    bh1, bh2, merged = get_bh_particles(ketju_file, tol)
    energy_mask = ketjugw.orbital_energy(bh1, bh2) > 0
    try:
        bound_idx = len(energy_mask) - get_idx_in_array(1, energy_mask[::-1])
        bound_state = "normal"
    except AssertionError:
        if np.all(energy_mask):
            _logger.logger.warning("Binary system has not become bound! We will use all data from the BH particles.")
            bound_idx = -1
            bound_state = "never"
        elif np.all(~energy_mask):
            bound_state = "always"
            _logger.logger.exception("Binary system is never unbound!", exc_info=True)
            raise
        else:
            _logger.logger.warning(f"Binary system is oscillating between bound and unbound, and data ends at an unbound point. We will use all data from the BH particles.")
            bound_idx = -1
            bound_state = "oscillate"
    return bh1[:bound_idx], bh2[:bound_idx], bound_state



def _do_linear_fitting(t, y, t0, tspan, return_idxs=False):
    """
    Wrapper around scipy's curve_fit, with start and end indices of the values 
    to fit from an array also calculated.

    Parameters
    ----------
    t : np.ndarray
        times, a subset of which the linear fit will be performed over
    y : np.ndarray
        corresponding y data, such that y = f(t)
    t0 : float
        time to begin the fit (must be same units as t)
    tspan : float
        duration over which the fit should be performed
    return_idxs : bool, optional
        return the array indices corresponding to t0 and t0+tspan, by default 
        False

    Returns
    -------
    popt : np.ndarray
        optimal parameter values [a,b], such that y = a*t+b
    idxs : list, optional
        indices corresponding to [t0, t0+tspan] if return_idxs is True
    """
    #determine index of t0 in t
    t0idx = np.argmax(t0 < t)
    tfidx = np.argmax(t0+tspan < t)
    #error when t0+tspan==t[-1]
    if t0+tspan >= t[-1]:
        tfidx = -1
        _logger.logger.warning("Analytical fit to binary evolution done to the end of the time data -> proceed with caution!")
    # assume dy/dt is a approx. linear
    popt, pcov = scipy.optimize.curve_fit(lambda x, a, b: a*x+b,
                                          t[t0idx:tfidx], y[t0idx:tfidx])
    if return_idxs:
        return popt, [t0idx, tfidx]
    else:
        return popt
 

def linear_fit_get_H(t, a, t0, tspan, Gps):
    """
    Determine the hardening rate H by performing a linear fit to the time 
    derivative of the inverse of the semimajor axis. The equation can be found
    in Merritt 2013, Eq, 8.15.

    Parameters
    ----------
    t : np.ndarray
        times corresponding to the orbital parameters [yr]
    a : np.ndarray
        semimajor axis values [pc]
    t0 : float
        time to start the linear fit
    tspan : float
        "duration" of linear fit
    Gps : float
        quantity G*rho/sigma in [1 / (pc * yr)]

    Returns
    -------
    : float
        hardening coefficient H
    """
    grad, c = _do_linear_fitting(t, 1/a, t0, tspan)
    return grad / Gps


def linear_fit_get_K(t, e, t0, tspan, H, Gps, a):
    """
    Determine the constant K in the evolution of eccentricity. The equation can 
    be found in Merritt 2013 Eq. 8.32, or in a nicer form in Mannerkoski 2019 
    Eq. 15. 

    Parameters
    ----------
    t : np.ndarray
        times corresponding to the orbital parameters [yr]
    e : np.ndarray
        eccentricity values
    t0 : float
        time to start the linear fit
    tspan : float
        "duration" of linear fit
    H : float
        hardening coefficient
    Gps : float
        constant G * density / sigma  [(pc * yr)^-1]
    a : np.ndarray
        semimajor axis values

    Returns
    -------
    : float
        eccentricity evolution constant K
    """
    popt, idxs = _do_linear_fitting(t, e, t0, tspan, return_idxs=True)
    delta_e = popt[0] * tspan # A * t1 + B - (A * t0 + B)
    integral_a = scipy.integrate.trapezoid(a[idxs[0]:idxs[1]], t[idxs[0]:idxs[1]])
    return delta_e / (Gps * H * integral_a) 


def analytic_evolve_peters_quinlan(a0, e0, t0, tf, m1, m2, Gps, H, K):
    """
    Analytically evolve a BH binary assuming hardening due to both stellar 
    scattering and GW emission

    Parameters
    ----------
    a0 : float
        initial semimajor axis
    e0 : float
        initial eccentricty
    t0 : float
        initial time of integration
    tf : float
        final time of integration
    m1 : float
        mass of particle 1
    m2 : float
        mass of particle 2
    Gps : float
        constant G * density / sigma  [(pc * yr)^-1]
    H : float
        hardening constant
    K : float
        eccentricity constant

    Returns
    -------
    tp : np.ndarray
        sampled integration times
    ap : np.ndarray
        integrated semimajor axis
    ep : np.ndarray
        integrated semimajor axis
    """
    #convert Gps to units used by ketjugw
    Gps = Gps / (ketjugw.units.pc * ketjugw.units.yr)

    def quinlan_derivatives(a, e, m1, m2):
        dadt = -a**2 * H * Gps
        dedt = -dadt / a * K
        return dadt, dedt

    propagate_time = 8*(tf-t0) + t0
    ap, ep, _,_, tp = ketjugw.orbit.peters_evolution(a0, e0, m1, m2, (t0, propagate_time, 5), ext_derivs=quinlan_derivatives)
    return tp, ap, ep


def get_hard_timespan(t, a, t_s, ah_s):
    """
    Determine for how long a binary is hard, e.g. the binary semimajor axis is 
    less than the hardening radius. Linear interpolation is used to find values of a between snapshot measurements.

    Parameters
    ----------
    t : array-like
        time where the binary is bound (E<0)
    a : array-like
        semimajor axis of the binary
    t_s : list
        time of snapshots where binary is bound (same units as t)
    ah_s : list
        semimajor axis of binary at snapshot times (same units as a)

    Returns
    -------
    float
        time duration where the binary is hard
    int
        array index corresponding to when the binary first becomes hard
    """
    f = scipy.interpolate.interp1d(t_s, ah_s, bounds_error=False, fill_value=(ah_s[0], ah_s[-1]))
    bool_arr = a < f(t)
    return np.sum(bool_arr) * (t[1]-t[0]), get_idx_in_array(1, bool_arr)


def find_idxs_of_n_periods(tval, tarr, sep, num_periods=1, max_iter=100, strict_mode=False):
    """
    Find the indices of a time series array corresponding to a given number of 
    periods about a desired time. Note the periods are taken to go from 
    pericentre to pericentre. Only odd-numbered num_periods values are 
    implemented for symmetry reasons, thus an even value of num_periods is the 
    same as calling the function with num_periods+1.
    The method can also be used for any approximately monotonic series `tarr`, 
    such as for example semimajor axis, provided the interval between data 
    points is reasonably short.

    Parameters
    ----------
    tval : float
        time value around which the number of periods will be determined
    tarr : array-like
        time array to search for tval within, must be of same units
    sep : array-like
        radial separation of BH binary as a time series
    num_periods : int, optional
        number of periods about tval to search for, by default 1
    max_iter : int, optional
        maximum number of allowed iterations in period search, be default 100
    strict_mode : bool, optional
        raise an error if insufficent number of periods are found, by default 
        False

    Returns
    -------
    idx : int
        index of tval in tarr
    end_idxs : list
        start and end indices of the orbital periods centred about the period 
        within which tval is
    """
    num_periods = int(num_periods)
    if num_periods%2 == 0:
        _logger.logger.warning(f"Only odd values of <num_periods> implemented, we will search for {num_periods+1} periods.")
    end_idxs = [0,0]
    # find the index of the time we want in the time series
    idx = get_idx_in_array(tval, tarr)
    # convert time series so pericentre passages have a value of 2 (apocentre 
    # have a value of -2)
    y = np.diff(np.sign(np.diff(sep)))
    found_peaks = False
    multiplier = 1
    max_idx = len(y)-1
    # gradually increase search bracket for efficiency
    iter_n = 0
    while not found_peaks:
        idxs = np.r_[max(0, idx-10*multiplier):min(max_idx, idx+10*multiplier)]
        peaks = np.where(y[idxs]==2)[0]
        _logger.logger.debug(f"Number of peaks: {len(peaks)}")
        if len(peaks) > 2*num_periods or iter_n == max_iter:
            if iter_n == max_iter:
                _logger.logger.error(f"Maximum number of iterations ({max_iter}) reached, and only {int(len(peaks)/2)}/{num_periods} have been found!")
                try:
                    assert not strict_mode
                except AssertionError:
                    _logger.logger.exception(f"Maximum iterations reached in determining orbital periods!", exc_info=True)
                    raise
            # have the number of orbits we want, return indices
            found_peaks = True
            peaks_rel = idxs[0]+peaks - idx
            # orbits are not of all same period, as orbit is shrinking
            # sometimes there are not num_orbits//2 orbits before the desired index, thus the first entry to end_idxs could be negative. Protect against this by ensuring the first index is always >=0, and truncating the number of periods used
            _idx0 = np.where(peaks_rel<0, peaks_rel, -np.inf).argmax()-num_periods//2
            if _idx0 < 0 :
                _idx1 = -_idx0
                _idx0 = 0
                try:
                    assert not strict_mode
                    _logger.logger.warning(f"Not enough complete orbits before desired time! Number of orbits used will be truncated to {_idx1}")
                except AssertionError:
                    _logger.logger.exception(f"Not enough complete orbits before desired time!", exc_info=True)
                    raise
            else:
                _idx1 = min(np.where(peaks_rel>=0, peaks_rel, np.inf).argmin()+num_periods//2, len(peaks_rel)-1)
            end_idxs[0] = peaks_rel[_idx0] + idx
            end_idxs[1] = peaks_rel[_idx1] + idx
            try:
                assert end_idxs[0] < end_idxs[1]
            except AssertionError:
                _logger.logger.exception(f"Period start index is greater than or equal to the period end index: {end_idxs}. An error has occurred in the calculation! Search mutlitplier was {multiplier}, central index was {idx}, and {len(peaks)} peaks have been identified.", 
                exc_info=True)
                raise
        else:
            # expand search bracket
            multiplier *= 2
            iter_n += 1
    return idx, end_idxs


def impact_parameter(bh1, bh2):
    """
    Determine the impact parameter as a function of time for two BHs

    Parameters
    ----------
    bh1 : ketjugw.Particle
        BH 1
    bh2 : ketjugw.Particle
        BH 2

    Returns
    -------
    b : array-like
        impact parameter
    theta : array-like
        angle between position vector and impact parameter vector
    """
    x = bh1.x - bh2.x
    v = bh1.v - bh2.v
    #v_hat = v / radial_separation(v)[:,np.newaxis]
    #b = x - v_hat * np.sum(x*v_hat, axis=-1)[:,np.newaxis]
    b = project_orthogonal(x,v)
    theta = angle_between_vectors(x,b)
    return b, theta


def move_to_centre_of_mass(bh1, bh2):
    """
    Determine centre of mass of two BHs. This implementation is taken from the
    ketjugw package.

    Parameters
    ----------
    bh1 : ketjugw.Particle
        BH 1
    bh2 : ketjugw.Particle
        BH 2

    Returns
    -------
    bh1 : ketjugw.Particle
        BH 1 in CoM frame
    bh2 : ketjugw.Particle
        BH 2 in CoM frame 
    """
    M = bh1.m + bh2.m
    x_CM = (bh1.m[:,np.newaxis]*bh1.x + bh2.m[:,np.newaxis]*bh2.x)/M[:,np.newaxis]
    v_CM = (bh1.m[:,np.newaxis]*bh1.v + bh2.m[:,np.newaxis]*bh2.v)/M[:,np.newaxis]
    bh1.x -= x_CM
    bh1.v -= v_CM
    bh2.x -= x_CM
    bh2.v -= v_CM
    # TODO way to edit in place?
    return bh1, bh2


def deflection_angle(bh1, bh2, peri_idx=None):
    """
    Determine the deflection angle due to scattering during a two-body encounter

    Parameters
    ----------
     bh1 : ketjugw.Particle
        BH 1
    bh2 : ketjugw.Particle
        BH 2
    peri_idx : int, optional
        indices of pericentre, by default None (calculates new)

    Returns
    -------
    array-like
        deflection angle (in radians)
    """
    M = bh1.m + bh2.m
    L = radial_separation(ketjugw.orbital_angular_momentum(bh1, bh2))
    E = ketjugw.orbital_energy(bh1, bh2)
    if peri_idx is None:
        _logger.logger.warning(f"Determining pericentre times using default inputs to `find_pericentre_time()`")
        _, peri_idx = find_pericentre_time(bh1, bh2)
    return 2 * np.arctan(M[peri_idx] / (L[peri_idx] * np.sqrt(2*E[peri_idx])))


def first_major_deflection_angle(angles, threshold=np.pi/2):
    """
    Determine the value of the first major deflection

    Parameters
    ----------
    angles : array-like
        deflection angles at pericentre
    threshold : float, optional
        minimum angle for a 'major' deflection, by default np.pi/2

    Returns
    -------
    : float
        first major deflection angle
    idx : int
        which deflection angle is the first major angle
    """
    if np.any(angles > threshold):
        idx = np.argmax(angles > threshold)
        return angles[idx], idx
    else:
        _logger.logger.warning(f"No deflection angles were greater than the threshold value of {threshold:.3f}! Largest is {np.max(angles):.3f}")
        return np.nan, None



#### CLASS DEFINITIONS THAT ARE NEEDED IN THIS FILE, AND SO SHOULD NOT ####
#### BE IN ANALYSIS_CLASSES.PY, TO PREVENT CIRCULAR IMPORTS            ####

class MergerInfo:
    def __init__(self):
        """
        A simple class to hold some information about the BH merger remnant
        """
        self._merged = False
        self._time = np.nan
        self._mass = np.nan
        self._kick = np.full(3, np.nan)
        self._spin = np.full(3, np.nan)
    
    def __call__(self):
        return self.merged
    
    def __str__(self):
        return f"BH Merger Remnant\n  Merged: {self.merged}\n  Time:   {self.time:<7.1f} Myr\n  Mass:   {self.mass:<7.1e} Msol\n  Kick:   {self.kick_magnitude:<7.1f} km/s\n  Chi:    {self.chi:<7.2f}"
    
    @property
    def merged(self):
        return self._merged

    @property
    def mass(self):
        return self._mass

    @property
    def time(self):
        return self._time

    @property
    def kick(self):
        return self._kick
    
    @property
    def kick_magnitude(self):
        return np.sqrt(np.sum(self.kick**2))

    @property
    def spin(self):
        return self._spin
    
    @property
    def spin_magnitude(self):
        return np.sqrt(np.sum(self.spin**2))
    
    @property
    def chi(self):
        return self.spin_magnitude/self.mass**2
    
    def update(self, bh):
        self._merged = True
        self._time = bh.t[0] / myr
        self._mass = bh.m[0]
        self._kick = bh.v / ketjugw.units.km_per_s
        self._spin = bh.spin[0,:]

