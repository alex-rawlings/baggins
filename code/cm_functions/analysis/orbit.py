import numpy as np
import scipy.spatial.distance, scipy.signal
import ketjugw.units
from ..mathematics import radial_separation

__all__ = ["find_pericentre_time"]

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
