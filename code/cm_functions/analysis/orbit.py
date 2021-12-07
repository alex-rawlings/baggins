import numpy as np
import scipy.spatial.distance, scipy.signal, scipy.optimize, scipy.integrate, scipy.interpolate
import ketjugw
import pygad
from .analyse_snap import _get_G_rho_per_sigma
from ..mathematics import radial_separation
from ..general import xval_of_quantity

__all__ = ["find_pericentre_time", "get_bh_particles", "get_bound_binary", "linear_fit_get_H", "linear_fit_get_K", "analytic_evolve_peters_quinlan"]

#common units
myr = 1e6 * ketjugw.units.yr
kpc = 1e3 * ketjugw.units.pc

def find_pericentre_time(bh1, bh2, height=-10, return_sep=False, **kwargs):
    """
    Determine the time (in Myr) of pericentre passages of the BHs with the 
    scipy.signals.find_peaks() function

    Parameters
    ----------
    bh1, bh2: ketjugw Particle object
    height: peaks required to have a height greater than this
    return_sep: return the positional-separation array
    kwargs: other keyword arguments for scipy.signal.find_peaks()

    Returns
    -------
    pericentre time: time in Myr of pericentre passages
    peak_idxs: index position of pericentre passage in BH Particle attribute
               arrays
    sep: (optional) the separation between the BHs as a function of time
    """
    sep = radial_separation(bh1.x/kpc, bh2.x/kpc)
    #pericentre is found by negating the separation, and identifying those 
    #peaks close to 0
    peak_idxs = scipy.signal.find_peaks(-sep, height=height, **kwargs)[0]
    if return_sep:
        return bh1.t[peak_idxs]/myr, peak_idxs, sep
    else:
        return bh1.t[peak_idxs]/myr, peak_idxs


def get_bh_particles(ketju_file, verbose=True):
    """
    Return the bh particles in the (usually-named) ketju_bhs.hdf5 file.
    This is really just a wrapper that ensures the bhs have the same time 
    domain.

    Parameters
    ----------
    ketju_file: path to ketju_bhs.hdf5 file to analyse
    verbose: verbose printing?

    Returns
    -------
    bh1, bh2: ketjugw.Particle objects for the BHs
    merged: bool, whether or not a merger occurred
    """
    bh1, bh2 = ketjugw.data_input.load_hdf5(ketju_file).values()
    len1, len2 = len(bh1.t), len(bh2.t)
    if len1 > len2:
        #bh2 has merged into bh1
        bh1 = bh1[:len2]
        merged = True
    elif len1 < len2:
        #bh1 has merged into bh2
        bh2 = bh2[:len1]
        merged = True
    else:
        #a merger has not occurred
        merged = False
    if verbose and merged:
        print("A merger has occurred.")
    return bh1, bh2, merged


def get_bound_binary(ketju_file, verbose=True):
    """
    Return the data from the ketju_bhs.hdf5 file corresponding to when the 
    binary becomes (and remains) bound. 

    Parameters
    ----------
    ketju_file: path to ketju_bhs.hdf5 file to analyse
    verbose: verbose printing?

    Returns
    -------
    bh1, bh2: ketjugw.Particle objects for the BHs, where the particles have the
              same time domain
    merged: bool, whether or not a merger occurred
    """
    bh1, bh2, merged = get_bh_particles(ketju_file, verbose)
    bhs = {0:bh1, 1:bh2}
    bh1, bh2 = list(ketjugw.find_binaries(bhs, remove_unbound_gaps=True).values())[0]
    return bh1, bh2, merged


def _do_linear_fitting(t, y, t0, tspan, return_idxs=False):
    """
    Wrapper around scipy's curve_fit, with start and end indices of the values 
    to fit from an array also calculated.

    Parameters
    ----------
    t: array of times, a subset of which the linear fit will be performed over
    y: corresponding y data, such that y = f(t)
    t0: time to begin the fit (must be same units as t)
    tspan: the duration over which the fit should be performed
    return_idxs (bool): return the array indices corresponding to t0 and
                        t0+tspan

    Returns
    -------
    popt: array of optimal parameter values [a,b], such that y = a*t+b
    idxs: list of indices corresponding to [t0, t0+tspan] if return_idxs is 
          True
    """
    #determine index of t0 in t
    t0idx = np.argmax(t0 < t)
    tfidx = np.argmax(t0+tspan < t)
    # assume dy/dt is a approx. linear
    popt, pcov = scipy.optimize.curve_fit(lambda x, a, b: a*x+b,
                                          t[t0idx:tfidx], y[t0idx:tfidx])
    if return_idxs:
        return popt, [t0idx, tfidx]
    else:
        return popt
 

def linear_fit_get_H(t, a, t0, tspan, snap, rh, return_Gps=False):
    """
    Determine the hardening rate H by performing a linear fit to the time 
    derivative of the inverse of the semimajor axis. The equation can be found
    in Merritt 2013, Eq, 8.15.

    Parameters
    ----------
    t: array of times corresponding to the orbital parameters in YEARS
    a: array of semimajor axis values in PC
    t0: time to start the linear fit
    tspan: "duration" of linear fit
    snap: pygad snapshot object from which velocity dispersion, density will
          be estimated
    rh: gravitational influence radius
    return_Gps (bool): return the value of G*rho/sigma for future use?

    Returns
    -------
    H: hardening coefficient
    G_rho_per_sigma: the so-named quantity, if return_Gps is True
    """
    assert(snap.phys_units_requested), "Snapshot must be given in physical units!"
    grad, c = _do_linear_fitting(t, 1/a, t0, tspan)
    if not isinstance(rh, pygad.UnitArr):
        print("Setting rh to default length units {}".format("pc"))
        rh = pygad.UnitScalar(rh, "pc")
    G_rho_per_sigma = _get_G_rho_per_sigma(snap, extent=rh)
    H = grad / G_rho_per_sigma
    if return_Gps:
        return H, G_rho_per_sigma
    else:
        return H


def linear_fit_get_K(t, e, t0, tspan, H, Gps, a):
    """
    Determine the constant K in the evolution of eccentricity. The equation can 
    be found in Merritt 2013 Eq. 8.32, or in a nicer form in Mannerkoski 2019 
    Eq. 15. 

    Parameters
    ----------
    t: array of times corresponding to the orbital parameters in YEARS
    e: array of eccentricity values 
    t0: time to start the linear fit
    tspan: "duration" of linear fit
    H: hardening coefficient
    Gps: constant G * density / sigma in (pc * yr)^-1
    a: array of semimajor axis values

    Returns
    -------
    eccentricity evolution constant K
    """
    popt, idxs = _do_linear_fitting(t, e, t0, tspan, return_idxs=True)
    delta_e = popt[0] * tspan # A * t1 + B - (A * t0 + B)
    integral_a = scipy.integrate.trapezoid(a[idxs[0]:idxs[1]], t[idxs[0]:idxs[1]])
    return delta_e / (Gps * H * integral_a) 


def analytic_evolve_peters_quinlan(a0, e0, t0, tf, m1, m2, Gps, H, K):
    #convert Gps to units used by ketjugw
    Gps = Gps / (ketjugw.units.pc * ketjugw.units.yr)

    def quinlan_derivatives(a, e, m1, m2):
        dadt = -a**2 * H * Gps
        dedt = -dadt / a * K
        return dadt, dedt

    propagate_time = 8*(tf-t0) + t0
    ap, ep, _,_, tp = ketjugw.orbit.peters_evolution(a0, e0, m1, m2, (t0, propagate_time, 5), ext_derivs=quinlan_derivatives)
    return tp, ap, ep
