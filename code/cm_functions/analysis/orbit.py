import warnings
import numpy as np
import scipy.spatial.distance, scipy.signal, scipy.optimize, scipy.integrate, scipy.interpolate
import ketjugw
from ..mathematics import radial_separation

__all__ = ["find_pericentre_time", "interpolate_particle_data", "get_bh_particles", "get_bound_binary", "linear_fit_get_H", "linear_fit_get_K", "analytic_evolve_peters_quinlan"]

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
        time of pericentre passages [Myr]
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
        return bh1.t[peak_idxs]/myr, peak_idxs, sep
    else:
        return bh1.t[peak_idxs]/myr, peak_idxs


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


def get_bh_particles(ketju_file, verbose=True, tol=1e-15):
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
    verbose : bool, optional
        verbose printing?, by default True
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
    #first need to determine if time series are consistent between particles
    if np.any(np.abs(bh1.t[:min_len] - bh2.t[:min_len])>tol):
        # particle time series are not in sync, need to interpolate
        # merger only occurs if Ketju is activated, in which case the time
        # series are in sync by construction, so no merger has occurred here
        # TODO is there a more robust way to ascertain if a merger has (not)
        # occurred that doesn't tie us to how Ketju data output occurs
        if verbose:
            print("Particle time series are not consistent with each other: linear interpolation will be performed")
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


def get_bound_binary(ketju_file, verbose=True, tol=1e-15):
    """
    Return the data from the ketju_bhs.hdf5 file corresponding to when the 
    binary becomes (and remains) bound. 

    Parameters
    ----------
    ketju_file : str
        path to ketju_bhs.hdf5 file to analyse
    verbose : bool, optional
        verbose printing?, by default True
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
    bh1, bh2, merged = get_bh_particles(ketju_file, verbose, tol)
    bhs = {0:bh1, 1:bh2}
    bh1, bh2 = list(ketjugw.find_binaries(bhs, remove_unbound_gaps=True).values())[0]
    return bh1, bh2, merged


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
        uration over which the fit should be performed
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
        warnings.warn("Analytical fit to binary evolution done to the end of the time data -> proceed with caution!")
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

