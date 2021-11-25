import numpy as np
import scipy.spatial.distance, scipy.signal
import ketjugw
from ..mathematics import radial_separation

__all__ = ["find_pericentre_time", "find_where_gw_dominate", "get_bh_particles", "get_bound_binary"]

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


def find_where_gw_dominate(orbit_params, err_level=0.1):
    """
    Find the index and time where GW emission becomes the dominant mechanism
    of binary hardening. Achieved by determining when the relative error 
    between the true orbital parameter evolution and the backwards-integrated 
    Peters evolution differ by more than err_level.
    """
    a, e, n, l = ketjugw.peters_evolution(orbit_params["a_R"][-1], orbit_params["e_t"][-1], orbit_params["m0"][0], orbit_params["m1"][0], -orbit_params["t"])
    a = a[::-1]
    e = e[::-1]
    n = n[::-1]
    l = l[::-1]
    err_a = np.abs((orbit_params["a_R"]-a)/orbit_params["a_R"])
    err_e = np.abs((orbit_params["e_t"]-e)/orbit_params["e_t"])
    idx = max(np.where(err_a>err_level)[0][-1], np.where(err_e>err_level)[0][-1])
    return idx, orbit_params["t"][idx]


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

