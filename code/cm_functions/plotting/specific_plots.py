import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


__all__ = ['plot_parameter_contours']


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
    print('Creating parameter contours...')
    if data_err is None:
        data_err = 0.3*yvals+1e-4
    for repeat in range(repeats):
        print('  Level: {:d}'.format(repeat+1))
        print('    Parameter 1 Limits: {:.2e} - {:.2e}'.format(init_lims[0][0], init_lims[0][1]))
        print('    Parameter 2 Limits: {:.2e} - {:.2e}'.format(init_lims[1][0], init_lims[1][1]))
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
    ax.legend(loc='upper left')
