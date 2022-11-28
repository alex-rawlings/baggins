import warnings
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import copy
import pygad
import ketjugw
from ..general import convert_gadget_time
from ..env_config import _cmlogger


__all__ = ["plot_galaxies_with_pygad", "GradientLinePlot", "GradientScatterPlot", "binary_param_plot", "twin_axes_from_samples", "voronoi_plot", "seaborn_jointplot_cbar", "draw_unit_sphere"]

_logger = _cmlogger.copy(__file__)


def plot_galaxies_with_pygad(snap, return_ims=False, orientate=None, figax=None, extent=None, kwargs=None, append_kwargs=False):
    """
    Convenience routine for plotting a system with pygad, both stars and DM

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to plot
    return_ims : bool, optional
        return list of images, by default False
    orientate : str, optional
        orientate the snapshot using pygad orientate_at method can be either to 
        an arbitrary vector, the angular momentum "L", or the semiminor axis of 
        the reduced intertia tensor "red I". If used, a shallow copy of the 
        snapshot is created, by default None
    figax : list, optional
        list of [fig, ax] from plt.subplots(), by default None
    extent : dict, optional
        dict of dicts with extent values with top-layer keys "stars" and "dm", 
        and second-layer keys "xz" and "xy",  e.g. extent["stars"]["xz"] = 100, 
        by default None
    kwargs : dict, optional
        other keyword arguments for the pygad plotting routine, by default None
    append_kwargs : bool, optional
        append given kwargs to default kwargs, by default False

    Returns
    -------
    fig : matplotlib.figure.Figure
        pyplot figure object
    ax : np.ndarray
        array of matplotlib.axes._subplots.AxesSubplot instances
    ims : list, optional
        list of images from pygad plotting routine
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
    
    """
    def __init__(self, ax, cmap="viridis", plot_kwargs={}):
        """
        Class to create pyplot plots with a colour gradient. The colour 
        gradient is consistent between all lines/points in the figure. This is 
        done by storing the data first, and then only plotting the data when 
        explicitly called.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            axis to plot to
        cmap : str, optional
            pyplot colour map name, by default "viridis"
        plot_kwargs : dict, optional
            arguments to be parsed to either plt.plot() or plt.scatter(), by 
            default {}
        """
        self.ax = ax
        self.data_count = 0
        self.all_x = []
        self.all_y = []
        self.all_c = []
        self.all_label = []
        self.cmap= getattr(plt.cm, cmap)
        self.all_marker = []
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
        x : np.ndarray
            x data
        y : np.ndarray
            y data
        c : np.ndarray
            data to map colours to
        label : _type_, optional
            label of plot, by default None
        marker : str, optional
            end marker, by default "o"
        plot_kwargs : dict, optional
            dict of other parameters to parse to pyplot.plot() or pyplot.scatter
            (), by default {}
        """
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_c.append(c)
        self.all_label.append(label)
        self.all_marker.append(marker)
        self.all_pks.append(plot_kwargs)
        self.data_count += 1
    
    def _set_colours(self, log=False):
        """
        Set the colours of the plot, should not be called directly

        Parameters
        ----------
        log : bool, optional
            colours in logscale?, by default False
        """
        vmin = min([np.nanmin(ci) for ci in self.all_c])
        vmax = max([np.nanmax(ci) for ci in self.all_c])
        if log:
            if vmin < 0:
                warnings.warn("Log scale normalisation cannot handle negative values! Using a linear scale")
                self.norm = colors.Normalize(vmin, vmax)
            else:
                self.norm = colors.LogNorm(vmin, vmax)
        else:
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
    def __init__(self, ax, cmap="viridis", plot_kwargs={}):
        super().__init__(ax, cmap=cmap, plot_kwargs=plot_kwargs)
    
    def plot(self, logcolour=False):
        """
        Plot the data, ensuring a consistent colour scheme.

        Parameters
        ----------
        logcolour : bool, optional
            colours in log scale?, by default False

        Raises
        ------
        ValueError
            no data to plot
        """
        if self.data_count < 1:
            raise ValueError("No data to plot!")
        self._set_colours(log=logcolour)
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
    
    def plot(self, logcolour=False):
        """
        Plot the data, ensuring a consistent colour scheme.

        Parameters
        ----------
        logcolour : bool, optional
            colours in log scale?, by default False

        Raises
        ------
        ValueError
            no data to plot
        """
        if self.data_count < 1:
            raise ValueError("No data to plot!")
        self._set_colours(log=logcolour)
        for xi, yi, ci, labeli, markeri, pki in zip(self.all_x, self.all_y, self.all_c, self.all_label, self.all_marker, self.all_pks):
            for i, (xs, ys, cs) in enumerate(zip(zip(xi[:-1], xi[1:]), zip(yi[:-1], yi[1:]), ci[:-1])):
                self.ax.scatter(xs, ys, color=self.cmap(self.norm(cs)), marker=markeri, label=(labeli if i==0 else ""),**pki)


def binary_param_plot(orbit_pars, ax=None, toffset=0, **kwargs):
    """
    Standard plot of binary semimajor axis and eccentricity.

    Parameters
    ----------
    orbit_pars : dict
        orbit parameters from ketjugw.orbital_parameters()
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axis to plot to, by default None (creates new instance)
    toffset : float, optional
        time offset, by default 0

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        plotting axis
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


def twin_axes_from_samples(ax, x1, x2, log=False):
    """
    Generate a twin axis for two discretely sampled quantities. Interpolation 
    is done between the samples for the plot axes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        parent axis to put the twin axis on
    y1 : array-like
        independent dataset 1, the 'original' variable
    y2 : array-like
        independent dataset 2, the 'transformed' variable
    log : bool, optional
        log scale for secondary axis?, by default False

    Returns
    -------
    ax2 : matplotlib.axes._subplots.AxesSubplot
        twin axis
    """
    try:
        assert np.all(np.diff(x1) > 0)
        assert np.all(np.sign(np.diff(x2)) == np.sign(x2[1]-x2[0]))
    except AssertionError:
        _logger.logger.exception(f"Original x-datasets must be strictly increasing!", exc_info=True)
        raise
    # set up forward and inverse functions with masked array handling
    def _forward(x):
        if isinstance(x, np.ma.MaskedArray):
            x = x.compressed()
        f = interp1d(x1, x2, bounds_error=False, fill_value="extrapolate")
        return f(x)
    def _inverse(x):
        if isinstance(x, np.ma.MaskedArray):
            x = x.compressed()
        f = interp1d(x2, x1, bounds_error=False, fill_value="extrapolate")
        return f(x)
    # set up secondary axis
    ax2 = ax.secondary_xaxis("top", functions=(_forward, _inverse))
    if log:
        ax2.set_xscale("log")
    return ax2


def voronoi_plot(vdat):
    """
    Plot the voronoi maps for a system.

    Parameters
    ----------
    vdat : dict
        voronoi values from analysis.voronoi_binned_los_V_statistics()
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
            norm = colors.CenteredNorm()
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
    adjust_kw : dict, optional
        dict to pass to pyplot.subplots_adjust describing how the subplot
        should be adjusted to accommodate the colorbar, by default {"top":0.9, 
        "bottom":0.1, "left":0.1, "right":0.8}
    cbarwidth : float, optional
        width of colorbar in axis units, by default 0.05
    cbargap : float, optional
        distance between marginal axis and colorbar in axis units, by default 
        0.02
    **kwargs : 
        keyword arguments to be parsed to sns.jointplot()

    Returns
    -------
    j : seaborn.JointGrid
        allow access to underlying axis
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
    """
    Draw a unit sphere

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        plotting axis
    points : int, optional
        number of points, by default 100
    """
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

