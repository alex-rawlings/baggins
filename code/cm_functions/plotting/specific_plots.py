from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import copy
import pygad
from ..general import convert_gadget_time


__all__ = ["plot_galaxies_with_pygad", "plot_parameter_contours"]


def plot_galaxies_with_pygad(snap, return_ims=False, orientate=None, figax=None, extent=None, kwargs=None):
    """
    Convenience routine for plotting a system with pygad, both stars and DM

    Parameters
    ----------
    snap: pygad snapshot to plot
    orientate: orientate the snapshot using pygad orientate_at method
               can be either to an arbitrary vector, the angular momentum 
               "L", or the semiminor axis of the reduced intertia tensor
               "red I". If used, a shallow copy of the snapshot is created
    extent: dict of extent values, with keys "stars" and "dm"
    scaleind: how to label the scale
    kwargs: dict of other keyword arguments for the pygad plotting routine

    Returns
    -------
    fig: pyplot figure object
    ax: subplot axes object as np.array
    """
    if orientate is not None:
        snap = copy.copy(snap)
        pygad.analysis.orientate_at(snap, orientate)
    if extent is None:
        extent = {"stars":None, "dm":None}
    if kwargs is None:
        kwargs = {"scaleind":"labels", "cbartitle":"", "Npx":800, "qty":"mass",
                    "fontsize":10}
    if figax is None:
        fig, ax = plt.subplots(2,2, figsize=(6, 6))
    else:
        fig, ax = figax[0], figax[1]
    ims = []
    time = convert_gadget_time(snap, new_unit="Myr")
    fig.suptitle("Time: {:.1f} Myr".format(time))
    ax[0,0].set_title("Stars")
    ax[0,1].set_title("DM Halo")
    for i in range(2):
        _,ax[i,0], imstars,*_ = pygad.plotting.image(snap.stars, xaxis=0, yaxis=2-i, extent=extent["stars"], ax=ax[i,0], **kwargs)
        _,ax[i,1], imdm,*_ = pygad.plotting.image(snap.dm, xaxis=0, yaxis=2-i, extent=extent["dm"], ax=ax[i,1], **kwargs)
        ims.append(imstars)
        ims.append(imdm)
    if return_ims:
        return fig, ax, ims
    else:
        return fig, ax


def plot_parameter_contours(ax, fun, xvals, yvals, init_lims, data_err=None, args=(), numPoints=350, repeats=1, slope=1.2, sigma_level=3):
    """
    Plot 2D parameter contours from chi2 test.
    # TODO: move chi2 calculation to mathematics submodule?

    Parameters
    ----------
    ax: the matplotlib axis to plot to
    fun: function to evaluate
    xvals: x coordinates to evaluate the function at
    yvals: y coordinates of data
    args: other arguments to the function to be evaluated as *args
    data_err: error in observed data
    init_limits: tuple of 2 arrays, each array with the min/max of a parameter
    numPoints: number of points to scan for each parameter
    repeats: allow for reducing search interval by zeroing in this many times
    slope: amount to decrease the search range by each repeat
    sigma_level: number of sigma levels to plot

    Returns
    -------
    None
    """
    assert(repeats > 0 and isinstance(repeats, int))
    print("Creating parameter contours...")
    if data_err is None:
        data_err = 0.3*yvals+1e-4
    for repeat in range(repeats):
        print("  Level: {:d}".format(repeat+1))
        print("    Parameter 1 Limits: {:.2e} - {:.2e}".format(init_lims[0][0], init_lims[0][1]))
        print("    Parameter 2 Limits: {:.2e} - {:.2e}".format(init_lims[1][0], init_lims[1][1]))
        param1_seq = np.linspace(init_lims[0][0], init_lims[0][1], numPoints)
        param2_seq = np.linspace(init_lims[1][0], init_lims[1][1], numPoints)

        #initialise chi2 array
        chi_array = np.full((numPoints, numPoints), np.nan)

        #scan the 2D parameter space
        for ind1, param1 in enumerate(param1_seq):
            for ind2, param2 in enumerate(param2_seq):
                fun_val = fun(xvals, param1, param2, *args)
                chi_array[ind1][ind2] = kfm.chi_square(yvals, fun_val, data_err)
        chimin = chi_array.min()
        pos_min = chi_array.argmin()
        pos_min = np.unravel_index(pos_min, (numPoints, numPoints))
        #reduce search range
        init_lims[0][0] = (1 - 1/slope**(repeat+1)) * param1_seq[pos_min[0]]
        init_lims[0][1] = (1 + 1/slope**(repeat+1)) * param1_seq[pos_min[0]]
        init_lims[1][0] = (1 - 1/slope**(repeat+1)) * param2_seq[pos_min[1]]
        init_lims[1][1] = (1 + 1/slope**(repeat+1)) * param2_seq[pos_min[1]]
    param1_min = param1_seq[pos_min[0]]
    param2_min = param2_seq[pos_min[1]]

    #determine sigma levels
    sigmas = np.full(sigma_level, np.nan)
    for s in range(1, sigma_level+1):
        ci = scipy.stats.chi2.cdf(s**2, 1)
        sigmas[s-1] = scipy.stats.chi2.ppf(ci, 1)

    #plot contour
    levels = sigmas + chimin
    ax.contourf(param2_seq, param1_seq, np.log10(chi_array), levels=50)
    ax.legend(loc="upper left")
