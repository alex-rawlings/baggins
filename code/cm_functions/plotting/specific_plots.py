import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats
import seaborn as sns
import copy
import pygad
import ketjugw
from ..general import convert_gadget_time
from .general import zero_centre_colour


__all__ = ["plot_galaxies_with_pygad", "GradientLinePlot", "GradientScatterPlot", "plot_parameter_contours", "binary_param_plot", "twin_axes_plot", "voronoi_plot", "seaborn_jointplot_cbar", "draw_unit_sphere"]


def plot_galaxies_with_pygad(snap, return_ims=False, orientate=None, figax=None, extent=None, kwargs=None, append_kwargs=False):
    """
    Convenience routine for plotting a system with pygad, both stars and DM

    Parameters
    ----------
    snap: pygad snapshot to plot
    orientate: orientate the snapshot using pygad orientate_at method
               can be either to an arbitrary vector, the angular momentum 
               "L", or the semiminor axis of the reduced intertia tensor
               "red I". If used, a shallow copy of the snapshot is created
    extent: dict of dicts with extent values with top-layer keys
            "stars" and "dm", and second-layer keys "xz" and "xy", 
            e.g. extent["stars"]["xz"] = 100
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
        extent = {"stars":{"xz":None, "xy":None}, "dm":{"xz":None, "xy":None}}
    default_kwargs = {"scaleind":"labels", "cbartitle":"", "Npx":800, 
                      "qty":"mass", "fontsize":10}
    if kwargs is None:
        kwargs = default_kwargs
    elif kwargs is not None and append_kwargs:
        kwargs = {**default_kwargs, **kwargs} #append some extra kwargs
    if figax is None:
        fig, ax = plt.subplots(2,2, figsize=(6, 6))
    else:
        fig, ax = figax[0], figax[1]
    ims = []
    time = convert_gadget_time(snap, new_unit="Myr")
    fig.suptitle("Time: {:.1f} Myr".format(time))
    ax[0,0].set_title("Stars")
    ax[0,1].set_title("DM Halo")
    for i, proj in enumerate(("xz", "xy")):
        _,ax[i,0], imstars,*_ = pygad.plotting.image(snap.stars, xaxis=0, yaxis=2-i, extent=extent["stars"][proj], ax=ax[i,0], **kwargs)
        _,ax[i,1], imdm,*_ = pygad.plotting.image(snap.dm, xaxis=0, yaxis=2-i, extent=extent["dm"][proj], ax=ax[i,1], **kwargs)
        ims.append(imstars)
        ims.append(imdm)
    if return_ims:
        return fig, ax, ims
    else:
        return fig, ax


class GradientPlot:
    """
    Class to create pyplot plots with a colour gradient. The colour 
    gradient is consistent between all lines/points in the figure. This is done 
    by storing the data first, and then only plotting the data when explicitly 
    called.
    """
    def __init__(self, ax, x, y, c, label=None, cmap="viridis", marker="o", plot_kwargs={}):
        """
        Initialise the plot with data.

        Parameters
        ----------
        ax: the axis to plot the figure to
        x: array of x data
        y: array of y data
        c: array to be used for gradient colouring (non-normalised values)
        label: label of plot
        cmap: pyplot colour map to use
        marker: end marker type (set to None to avoid)
        plot_kwargs: dict of other parameters to parse to pyplot.plot()
        """
        self.ax = ax
        self.data_count = 1
        self.all_x = [x]
        self.all_y = [y]
        self.all_c = [c]
        self.all_label = [label]
        self.cmap= getattr(plt.cm, cmap)
        self.all_marker = [marker]
        self.all_pks = [plot_kwargs]
        self.norm = [0,1]
    
    def __len__(self):
        return self.data_count

    def add_data(self, x, y, c, label=None, marker="o", plot_kwargs={}):
        """
        Add a dataset to the plot (note that the data is just stored here for 
        future use).

        Parameters
        ----------
        see comments for __init__
        """
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_c.append(c)
        self.all_label.append(label)
        self.all_marker.append(marker)
        self.all_pks.append(plot_kwargs)
        self.data_count += 1
    
    def set_colours(self):
        vmin = min([min(ci) for ci in self.all_c])
        vmax = max([max(ci) for ci in self.all_c])
        self.norm = colors.Normalize(vmin, vmax)
    
    def add_cbar(self, **kwargs):
        """
        Add a colour bar to the plot.
        """
        plt.colorbar(plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), ax=self.ax, **kwargs)

    def add_legend(self, **kwargs):
        """
        Add a legend to the plot.
        """
        self.ax.legend(**kwargs)


class GradientLinePlot(GradientPlot):
    """
    Apply the GradientPlot class for pyplot line plots
    """
    def __init__(self, ax, x, y, c, label=None, cmap="viridis", marker="o", plot_kwargs={}):
        super().__init__(ax, x, y, c, label=label, cmap=cmap, marker=marker, plot_kwargs=plot_kwargs)
    
    def plot(self):
        """
        Plot the data, ensuring a consistent colour scheme.
        """
        self.set_colours()
        for xi, yi, ci, labeli, markeri, pki in zip(self.all_x, self.all_y, self.all_c, self.all_label, self.all_marker, self.all_pks):
            if markeri is not None:
                self.ax.scatter(xi[-1], yi[-1], color=self.cmap(self.norm(ci[-1])), marker=markeri, label=labeli, zorder=10*self.data_count)
            for xs, ys, cs in zip(zip(xi[:-1], xi[1:]), zip(yi[:-1], yi[1:]), ci[:-1]):
                self.ax.plot(xs, ys, color=self.cmap(self.norm(cs)), **pki)


class GradientScatterPlot(GradientPlot):
    """
    Apply the GradientPlot class for pyplot scatter plots
    """
    def __init__(self, ax, x, y, c, label=None, cmap="viridis", marker="o", plot_kwargs={}):
        super().__init__(ax, x, y, c, label=label, cmap=cmap, marker=marker, plot_kwargs=plot_kwargs)
    
    def plot(self):
        """
        Plot the data, ensuring a consistent colour scheme.
        """
        self.set_colours()
        for xi, yi, ci, labeli, markeri, pki in zip(self.all_x, self.all_y, self.all_c, self.all_label, self.all_marker, self.all_pks):
            for i, (xs, ys, cs) in enumerate(zip(zip(xi[:-1], xi[1:]), zip(yi[:-1], yi[1:]), ci[:-1])):
                self.ax.scatter(xs, ys, color=self.cmap(self.norm(cs)), marker=markeri, label=(labeli if i==0 else ""),**pki)


def binary_param_plot(orbit_pars, ax=None, toffset=0, **kwargs):
    """
    Standard plot of binary semimajor axis and eccentricity.

    Parameters
    ----------
    orbit_pars: orbit parameter dictionary
    ax: matplotlib axis object to add plot to, None to create a new instance
    toffset: time offset in Myr
    kwargs: arguments to be parsed to pyplot.plot

    Returns
    -------
    ax: matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(2,1,sharex="col")
    ax[0].set_ylabel("a/pc")
    ax[1].set_ylabel("e")
    ax[1].set_xlabel("t/Myr")
    ax[1].set_ylim(0,1)
    myr = ketjugw.units.yr * 1e6
    ax[0].semilogy(orbit_pars["t"]/myr + toffset, orbit_pars["a_R"]/ketjugw.units.pc, **kwargs)
    ax[1].plot(orbit_pars["t"]/myr + toffset, orbit_pars["e_t"], **kwargs)
    return ax


class twin_axes_plot:
    def __init__(self, ax, convert_func, share="y"):
        """
        Set up a shared axis for a plot. An example would be time and redshift
        on the top and bottom x-axes.
        """
        self.ax = ax
        self.share = share
        if share == "y":
            self.twin_ax = ax.twiny()
        else:
            self.twin_ax = ax.twinx()
        self.convert_func = convert_func
        self.ax.callbacks.connect("ylim_changed", self.converter)
    
    def converter(self, ax):
        if self.share == "x":
            y1, y2 = ax.get_ylim()
            self.twin_ax.set_ylim(self.convert_func(y1), self.convert_func(y2))
        else:
            x1, x2 = ax.get_xlim()
            self.twin_ax.set_xlim(self.convert_func(x1), self.convert_func(x2))
        self.twin_ax.figure.canvas.draw()


def voronoi_plot(vdat):
    """
    Plot the voronoi maps for a system.

    Parameters
    ----------
    vdat: dict of voronoi values from analysis.voronoi_binned_los_V_statistics()
    """
    fig, ax = plt.subplots(2,2, sharex="all", sharey="all", figsize=(7,4.7))
    ax[0,0].set_ylabel("y/kpc")
    ax[1,0].set_xlabel("x/kpc")
    ax[1,0].set_ylabel("y/kpc")
    ax[1,1].set_xlabel("x/kpc")
    ax = np.concatenate(ax).flat
    for i, (stat, cmap, label) in enumerate(zip(
        (vdat["img_V"], vdat["img_sigma"], vdat["img_h3"], vdat["img_h4"]),
        ("seismic", "plasma", "seismic", "seismic"),
        (r"$V$ [km/s]", r"$\sigma$ [km/s]", r"$h_3$ [km/s]", r"$h_4$ [km/s]")
    )):
        #plot the statistic
        if i != 1:
            norm = zero_centre_colour(stat)
        else:
            norm = colors.Normalize(stat.min(), stat.max())
        ax[i].set_aspect("equal")
        p1 = ax[i].imshow(stat, interpolation="nearest", origin="lower", extent=vdat["extent"], cmap=cmap, norm=norm)
        cbar = plt.colorbar(p1, ax=ax[i])
        cbar.ax.set_ylabel(label)


def seaborn_jointplot_cbar(adjust_kw={"top":0.9, "bottom":0.1, "left":0.1, "right":0.8}, cbarwidth=0.05, cbargap=0.02, **kwargs):
    """
    Wrapper to add a colorbar to a seaborn jointplot() object.

    Parameters
    ----------
    adjust_kw: dict to pass to pyplot.subplots_adjust describing how the subplot
               should be adjusted to accommodate the colorbar
    cbarwidth: width of colorbar in axis units
    cbargap: distance between marginal axis and colorbar in axis units
    **kwargs: keyword arguments to be parsed to sns.jointplot()

    Returns
    -------
    j: seaborn.JointGrid object
    """
    j = sns.jointplot(**kwargs)
    plt.subplots_adjust(**adjust_kw)
    #get current positions of axes
    pos_joint_ax = j.ax_joint.get_position()
    pos_margx_ax = j.ax_marg_x.get_position()
    #reposition the joint ax so it has the same width as margx
    j.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_margx_ax.width, pos_joint_ax.height])
    #reposition colorbar
    j.figure.axes[-1].set_position([adjust_kw["right"]+cbargap, pos_joint_ax.y0, cbarwidth, pos_joint_ax.height])
    return j


def draw_unit_sphere(ax, points=100):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_box_aspect((2,2,2))
    theta = np.linspace(0, np.pi, int(points/2))[:-1]
    phi = np.linspace(0, 2*np.pi, points)[:-1]
    u, v = np.meshgrid(theta, phi)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax.plot_wireframe(x, y, z, alpha=0.2, color='k')
    #plot origin
    ax.scatter(0,0,0, color='k')


###############################
######## NEEDS UPDATING #######
###############################
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
