import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import copy
import pygad
import ketjugw
from ..general import convert_gadget_time, units
from ..env_config import _cmlogger


__all__ = ["plot_galaxies_with_pygad", "binary_param_plot", "twin_axes_from_samples", "voronoi_plot", "seaborn_jointplot_cbar", "draw_unit_sphere", "heatmap", "annotate_heatmap"]

_logger = _cmlogger.getChild(__name__)


def plot_galaxies_with_pygad(snap, return_ims=False, orientate=None, figax=None, extent=None, kwargs=None, append_kwargs=False, black_background=True):
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
    black_background : bool, optional
        set background colour to black, by default False

    Returns
    -------
    fig : matplotlib.figure.Figure
        pyplot figure object
    ax : np.ndarray
        array of matplotlib.axes.Axes instances
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
        try:
            _,ax[i,0], imstars,*_ = pygad.plotting.image(snap.stars, xaxis=0, yaxis=2-i, extent=extent["stars"][proj], ax=ax[i,0], **kwargs)
            ims.append(imstars)
        except RuntimeError:
            _logger.error("No stars in snapshot!")
        try:
            _,ax[i,1], imdm,*_ = pygad.plotting.image(snap.dm, xaxis=0, yaxis=2-i, extent=extent["dm"][proj], ax=ax[i,1], **kwargs)
            ims.append(imdm)
        except RuntimeError:
            _logger.error("No DM in snapshot!")
    if black_background:
        _logger.debug("Setting black background")
        for axi in np.concatenate(ax): axi.set_facecolor("k")
    if return_ims:
        return fig, ax, ims
    else:
        return fig, ax


def binary_param_plot(orbit_pars, ax=None, toffset=0, **kwargs):
    """
    Standard plot of binary semimajor axis and eccentricity.

    Parameters
    ----------
    orbit_pars : dict
        orbit parameters from ketjugw.orbital_parameters()
    ax : matplotlib.axes.Axes, optional
        axis to plot to, by default None (creates new instance)
    toffset : float, optional
        time offset, by default 0

    Returns
    -------
    ax : matplotlib.axes.Axes
        plotting axis
    """
    if ax is None:
        fig, ax = plt.subplots(3,1,sharex="col")
    ax[0].set_ylabel("a/pc")
    ax[1].set_ylabel("e")
    ax[2].set_ylabel("1-e")
    ax[-1].set_xlabel("t/Myr")
    ax[1].set_ylim(0,1)
    ax[0].semilogy(orbit_pars["t"]/units.Myr + toffset, orbit_pars["a_R"]/ketjugw.units.pc, **kwargs)
    ax[1].plot(orbit_pars["t"]/units.Myr + toffset, orbit_pars["e_t"], **kwargs)
    ax[2].semilogy(orbit_pars["t"]/units.Myr + toffset, 1-orbit_pars["e_t"], **kwargs)
    return ax


def twin_axes_from_samples(ax, x1, x2, log=False):
    """
    Generate a twin axis for two discretely sampled quantities. Interpolation 
    is done between the samples for the plot axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        parent axis to put the twin axis on
    x1 : array-like
        independent dataset 1, the 'original' variable
    x2 : array-like
        independent dataset 2, the 'transformed' variable
    log : bool, optional
        log scale for secondary axis?, by default False

    Returns
    -------
    ax2 : matplotlib.axes.Axes
        twin axis
    """
    try:
        assert np.all(np.diff(x1) > 0)
        assert np.all(np.sign(np.diff(x2)) == np.sign(x2[1]-x2[0]))
    except AssertionError:
        _logger.exception(f"Original x-datasets must be strictly increasing!", exc_info=True)
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
    ax : matplotlib.axes.Axes
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


def heatmap(data, row_labels, col_labels, ax=None, cmap="cividis", show_cbar=True, cbar_kw=None, cbarlabel="w", bad_colour="w", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels, taken from:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html 

    Parameters
    ----------
    data : np.ndarraay
        array of shape (M, N)
    row_labels : list-like
        list or array of length M with the labels for the rows
    col_labels : list-like
        list or array of length N with the labels for the columns
    ax : matplotlib.axes.Axes, optional
        axis to which the heatmap is plotted, by default None
    cbar_kw : dict, optional
        arguments to `matplotlib.Figure.colorbar`, by default None
    cbarlabel : str, optional
        label for the colorbar, by default ""
    **kwargs
        All other arguments are forwarded to `imshow`
    
    Returns
    -------
    im : matplotlib.image.AxesImage
        image of data
    ax : matplotlib.axes.Axes
        plotting axes
    cbar : matplotlib.colorbar
        colourbar
    """

    if ax is None:
        fig, ax = plt.subplots(1,1)
    if cbar_kw is None:
        cbar_kw = {}
    try:
        cmapv = getattr(plt.cm, cmap)
    except AttributeError:
        _logger.warning(f"{cmap} does not exist. Using default colormap: cividis")
        cmapv = plt.cm.cividis
    # set bad grids to a specified colour
    cmapv.set_bad(color=bad_colour)
    # plot the heatmap
    im = ax.imshow(data, cmap=cmapv, **kwargs)
    cbar = None
    if show_cbar:
        # create colorbar
        cbar = plt.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=90)

    # show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left", va="top", rotation_mode="anchor")

    # turn spines off and create white grid
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, ax, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap, taken from:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html  

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        AxesImage to be labeled
    data : array-like, optional
        data used to annotate (if None, the image's data is used), by default 
        None
    valfmt : str, optional
        format of the annotations inside the heatmap, should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`, by default "{x:.2f}"
    textcolors : list-like, optional
        pair of colors, with the first used for values below a threshold,
        the second for those above, by default ("black", "white")
    threshold : float, optional
        value in data units according to which the colors from textcolors are
        applied, by default None (uses the middle of the colormap as
        separation)
    **kwargs
        all other arguments are forwarded to each call to `text` used to create
        the text labels
    
    Returns
    texts : list
        text in each pixel
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.nanmax(data))/2.

    # set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # loop over the data and create a `Text` for each "pixel".
    # change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if isinstance(data[i,j], np.ma.core.MaskedConstant):
                continue
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

