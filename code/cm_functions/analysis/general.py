import numpy as np
import scipy.stats
import warnings
import pygad
from ..general import convert_gadget_time
from ..mathematics import get_histogram_bin_centres


__all__ = ["beta_profile", "snap_num_for_time"]


def beta_profile(r, vspherical, binwidth, qcut=0.98, logbin=True, eps=1e-16):
    """
    Determine the beta profile as defined in B&T Eq. 4.61
    
    Parameters
    ----------
    r: array of radial positions
    vspherical: (n,3) array  the spherical velocity components, with columns
                corresponding to radius, theta, and phi velocities
    binwidth: fixed width of bins (dex for logscale)
    qcut: remove those particles which are greater than qcut quantile 
          (e.g. those few particles that are very far away)
    logbin: binning done equally in a logarithmic scale
    eps: small number to prevent division by 0
    
    Returns
    -------
    beta: beta profile binned into nbin radial bins
    bin_centres: central value of each radial bin 
    bincounts: number of particles within each radial bin
    """
    #create the radial mask if required
    if qcut < 1:
        mask = r < np.quantile(r, qcut)
        r = r[mask]
        vspherical = vspherical[mask, :]
    #determine the bins -> used fixed binwidths
    if logbin:
        bins = 10**np.arange(-3, np.log10(np.max(r))+binwidth, binwidth)
    else:
        bins = np.arange(0, np.max(r)+binwidth, binwidth)
    #bin the statistics
    standard_devs, bin_edges, binnumbers = scipy.stats.binned_statistic(r, [vspherical[:,0], vspherical[:,1], vspherical[:,2]], statistic="std", bins=bins)
    #mask out nan values
    nanmask = ~np.any(np.isnan(standard_devs), axis=0)
    standard_devs = standard_devs[:, nanmask]
    #get the counts in each bin
    bincounts,*_ = np.histogram(r, bins=bins)
    #calculate beta(r)
    beta = 1 - (standard_devs[1,:]**2 + standard_devs[2,:]**2) / (2 * standard_devs[0,:]**2+eps)
    bin_centres = get_histogram_bin_centres(bin_edges)
    bin_centres = bin_centres[nanmask]
    bincounts = bincounts[nanmask]
    return beta, bin_centres, bincounts


def snap_num_for_time(snaplist, time_to_find, units="Myr", method="floor"):
    """
    Determine the snapshot number for the given time. May result in the
    last snapshot in the list to be returned if the given time is much later
    than the snapshots.

    Parameters
    ----------
    snaplist: list of snapshot files
    time_to_find: time we want to find
    units: units of the time, default Myr
    method: one of
                - 'floor': last snapshot before the given time
                - 'ceil': first snapshot after the given time
                - 'nearest': snapshot closest to the given time

    Returns
    -------
    idx: index in the list of snapshots which is closest to (but earlier than)
         the sought time 
    """
    if method not in ["floor", "ceil", "nearest"]: raise ValueError("method must be one of 'floor', 'ceil', or 'nearest'.")
    assert(isinstance(time_to_find, float) or isinstance(time_to_find, int))
    for ind, this_snap in enumerate(snaplist):
        snap = pygad.Snapshot(this_snap, physical=True)
        this_time = convert_gadget_time(snap, new_unit=units)
        snap.delete_blocks()
        del snap
        if ind == 0:
            prev_time = this_time
            continue
        if time_to_find < this_time:
            if method == "floor":
                idx = ind-1
            elif method == "ceil":
                idx = ind
            else:
                if this_time-time_to_find > time_to_find-prev_time:
                    idx = ind-1 #closest snap is the one before
                else:
                    idx = ind #closest snap is the one after
            break
        prev_time = this_time
    else:
        idx = len(snaplist)-1
        warnings.warn("Returning the final snapshot in the list!")
    return idx
