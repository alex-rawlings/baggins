import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import copy
import pygad
import ketjugw
from ..general import convert_gadget_time, units
from ..env_config import _cmlogger


__all__ = [
    "plot_galaxies_with_pygad",
    "binary_param_plot",
    "twin_axes_from_samples",
    "voronoi_plot",
    "seaborn_jointplot_cbar",
    "draw_unit_sphere",
    "heatmap",
    "annotate_heatmap",
    "violinplot",
]

_logger = _cmlogger.getChild(__name__)


def plot_galaxies_with_pygad(
    snap,
    return_ims=False,
    orientate=None,
    ax=None,
    extent=None,
    kwargs=None,
    overplot_bhs=False,
):
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
    ax : matplotlib.axes.Axes, optional
        axis to plot to, by default None (creates new instance)
    extent : dict, optional
        dict of dicts with extent values with top-layer keys "stars", "dm", "gas",
        and second-layer keys "xz" and "xy",  e.g. extent["stars"]["xz"] = 100,
        by default None
    kwargs : dict, optional
        other keyword arguments for the pygad plotting routine, by default None
    overplot_bhs : bool, optional
        plot BHs with a white point, by default False

    Returns
    -------
    fig : matplotlib.figure.Figure
        pyplot figure object
    ax : np.ndarray
        array of matplotlib.axes.Axes instances
    ims : list, optional
        list of images from pygad plotting routine
    """
    fams = [pt for pt in ("stars", "dm", "gas") if pt in snap.families()]
    if orientate is not None:
        snap = copy.copy(snap)
        snap.to_physical_units()
        pygad.analysis.orientate_at(snap, orientate)
    _extent = {
        "stars": {"xz": None, "xy": None},
        "dm": {"xz": None, "xy": None},
        "gas": {"xz": None, "xy": None},
    }
    if _extent is not None:
        _extent.update(extent)
    default_kwargs = {
        "scaleind": "labels",
        "cbartitle": "",
        "Npx": 800,
        "qty": "mass",
        "fontsize": 10,
    }
    if kwargs is None:
        kwargs = default_kwargs
    else:
        kwargs = {**default_kwargs, **kwargs}  # append some extra kwargs
    if ax is None:
        fig, ax = plt.subplots(
            2, len(fams), figsize=(3 * len(fams), 6), sharex="col", squeeze=False
        )
    else:
        fig = ax.get_figure()
    ims = []
    time = convert_gadget_time(snap, new_unit="Myr")
    fig.suptitle(f"Time: {time:.1f} Myr")
    for i, pt in enumerate(fams):
        _logger.info(f"Plotting {pt}")
        ax[0, i].set_title(pt)
        for j, proj in enumerate(("xz", "xy")):
            _, ax[j, i], im, *_ = pygad.plotting.image(
                getattr(snap, pt),
                xaxis=0,
                yaxis=2 - j,
                extent=_extent[pt][proj],
                ax=ax[j, i],
                **kwargs,
            )
            ims.append(im)
            if overplot_bhs:
                ax[j, i].scatter(snap.bh["pos"][:, 0], snap.bh["pos"][:, 2 - j], c="w")
    for axi in np.concatenate(ax):
        axi.set_facecolor("k")
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
        fig, ax = plt.subplots(3, 1, sharex="col")
    ax[0].set_ylabel("a/pc")
    ax[1].set_ylabel("e")
    ax[2].set_ylabel("1-e")
    ax[-1].set_xlabel("t/Myr")
    ax[1].set_ylim(0, 1)
    ax[0].semilogy(
        orbit_pars["t"] / units.Myr + toffset,
        orbit_pars["a_R"] / ketjugw.units.pc,
        **kwargs,
    )
    ax[1].plot(orbit_pars["t"] / units.Myr + toffset, orbit_pars["e_t"], **kwargs)
    ax[2].semilogy(
        orbit_pars["t"] / units.Myr + toffset, 1 - orbit_pars["e_t"], **kwargs
    )
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
        assert np.all(np.sign(np.diff(x2)) == np.sign(x2[1] - x2[0]))
    except AssertionError:
        _logger.exception(
            "Original x-datasets must be strictly increasing!", exc_info=True
        )
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


def voronoi_plot(vdat, ax=None, figsize=(7, 4.7), clims={}, desat=False):
    """
    Plot the voronoi maps for a system.

    Parameters
    ----------
    vdat : dict
        voronoi values from analysis.voronoi_binned_los_V_statistics()
    ax : np.ndarray, optional
        numpy array of pyplot.Axes objects for plotting, by default None
    figsize : tuple, optional
        figure size, by default (7,4.7)
    clims : dict, optional
        colour scale limits, by default None
    desat : bool, optional
        use a desaturated colour scheme, by default False

    Returns
    -------
    ax : np.ndarray
        plotting axes
    """
    # set the colour limits
    _clims = dict(V=[None], sigma=[None, None], h3=[None], h4=[None])
    for k, v in clims.items():
        try:
            assert isinstance(v, (list, tuple))
        except AssertionError:
            _logger.exception(
                f"Each value of `clim` must be a list or tuple, not {type(v)}!",
                exc_info=True,
            )
            raise
        vlen = 2 if k == "sigma" else 1
        try:
            assert len(v) == vlen
        except AssertionError:
            _logger.exception(
                f"`clim` entry for {k} must be of length {vlen}, not {len(v)}!",
                exc_info=True,
            )
            raise
        _clims[k] = v

    # set up the figure
    if ax is None:
        fig, ax = plt.subplots(2, 2, sharex="all", sharey="all", figsize=figsize)
        for i in range(2):
            ax[1, i].set_xlabel(r"$x/\mathrm{kpc}$")
            ax[i, 0].set_ylabel(r"$y/\mathrm{kpc}$")
    if desat:
        div_cols = colormaps.get("voronoi_div_desat")
        asc_cols = colormaps.get("voronoi_seq_desat")
    else:
        div_cols = colormaps.get("voronoi_div")
        asc_cols = colormaps.get("voronoi_seq")
    for i, (statkey, axi, cmap, label) in enumerate(
        zip(
            ("V", "sigma", "h3", "h4"),
            ax.flat,
            (div_cols, asc_cols, div_cols, div_cols),
            (
                r"$V/\mathrm{km}\,\mathrm{s}^{-1}$",
                r"$\sigma/\mathrm{km}\,\mathrm{s}^{-1}$",
                r"$h_3$",
                r"$h_4$",
            ),
        )
    ):
        # plot the statistic
        cmap.set_bad(color="k")
        stat = vdat[f"img_{statkey}"]
        if i != 1:
            norm = colors.CenteredNorm(vcenter=0, halfrange=_clims[statkey][0])
        else:
            if _clims["sigma"]:
                norm = colors.Normalize(*_clims["sigma"])
            else:
                norm = colors.Normalize(stat.min(), stat.max())
        axi.set_aspect("equal")
        p1 = axi.imshow(
            stat,
            interpolation="nearest",
            origin="lower",
            extent=vdat["extent"],
            cmap=cmap,
            norm=norm,
        )
        divider = make_axes_locatable(axi)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(p1, cax=cax)
        cbar.ax.set_ylabel(label)
    return ax


def seaborn_jointplot_cbar(
    adjust_kw={"top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.8},
    cbarwidth=0.05,
    cbargap=0.02,
    **kwargs,
):
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
    # get current positions of axes
    pos_joint_ax = j.ax_joint.get_position()
    pos_margx_ax = j.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as margx
    j.ax_joint.set_position(
        [pos_joint_ax.x0, pos_joint_ax.y0, pos_margx_ax.width, pos_joint_ax.height]
    )
    # reposition colorbar
    j.figure.axes[-1].set_position(
        [adjust_kw["right"] + cbargap, pos_joint_ax.y0, cbarwidth, pos_joint_ax.height]
    )
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((2, 2, 2))
    theta = np.linspace(0, np.pi, int(points / 2))[:-1]
    phi = np.linspace(0, 2 * np.pi, points)[:-1]
    u, v = np.meshgrid(theta, phi)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax.plot_wireframe(x, y, z, alpha=0.2, color="k")
    # plot origin
    ax.scatter(0, 0, 0, color="k")


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cmap="cividis",
    show_cbar=True,
    cbar_kw=None,
    cbarlabel="w",
    bad_colour="w",
    **kwargs,
):
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
        fig, ax = plt.subplots(1, 1)
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
    plt.setp(
        ax.get_xticklabels(), rotation=-30, ha="left", va="top", rotation_mode="anchor"
    )

    # turn spines off and create white grid
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, ax, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
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
        threshold = im.norm(np.nanmax(data)) / 2.0

    # set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # loop over the data and create a `Text` for each "pixel".
    # change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if isinstance(data[i, j], np.ma.core.MaskedConstant):
                continue
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


def violinplot(d, pos=None, ax=None, showbox=True, lcol=None, boxwidth=5, **kwargs):
    """
    Generate a violin plot with an inner box and whisker plot

    Parameters
    ----------
    d : list-like
        data to create plot for. If a 2D array is given, violins are for each
        column
    pos : list-like, optional
        x coordinates of data, by default None
    ax : matplotlib.axes.Axes, optional
        plotting axes, by default None
    showbox : bool, optional
        show the IQR box, by default True
    lcol : str, optional
        line colour, by default None
    boxwidth : float, optional
        width of IQR box, by default 5

    Returns
    -------
    ax : matplotlib.axes.Axes, optional
        plotting axes
    """

    def _adjacent_vals(vals, q1, q3):
        # helper function to determine whisker limits
        # follows same limit convention as pyplot.boxplot()
        vals = vals[~np.isnan(vals)]
        max_val = max(vals)
        min_val = min(vals)
        upper_av = q3 + (q3 - q1) * 1.5
        try:
            assert q3 <= max_val
        except AssertionError:
            _logger.exception(f"{q3} must be less than {max_val}", exc_info=True)
            raise
        upper_av = np.clip(upper_av, q3, max_val)
        lower_av = q1 - (q3 - q1) * 1.5
        try:
            assert min_val <= q1
        except AssertionError:
            _logger.exception(f"{min_val} must be less than {q1}", exc_info=True)
            raise
        lower_av = np.clip(lower_av, min_val, q1)
        return lower_av, upper_av

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    lcol = "#373737" if lcol is None else lcol

    # determine the whiskers
    quartile1, medians, quartile3 = np.nanquantile(d, [0.25, 0.5, 0.75], axis=-1)
    whiskers = np.array(
        [
            _adjacent_vals(sorted_array, q1, q3)
            for sorted_array, q1, q3, in zip(d, quartile1, quartile3)
        ]
    )
    whisker_min, whisker_max = whiskers[:, 0], whiskers[:, 1]

    # add whiskers and median, truncate the data for violins
    if showbox:
        ax.scatter(pos, medians, marker="o", color="white", s=boxwidth, zorder=3)
        ax.vlines(pos, quartile1, quartile3, color=lcol, ls="-", lw=boxwidth, zorder=1)
        ax.vlines(
            pos,
            whisker_min,
            whisker_max,
            color=lcol,
            ls="-",
            lw=0.2 * boxwidth,
            zorder=2,
        )
    trunc_d = []
    for dd, wmin, wmax in zip(d, whisker_min, whisker_max):
        mask = np.logical_and(dd > wmin, dd < wmax)
        trunc_d.append(np.array(dd)[mask])

    if "widths" in kwargs and isinstance(kwargs["widths"], (float, int)):
        kwargs["widths"] = np.repeat(kwargs["widths"], len(pos))
    kwargs["showmedians"] = False if showbox else True
    violins = ax.violinplot(
        trunc_d, positions=pos, showmeans=False, showextrema=False, **kwargs
    )

    # update styling of violin
    for i, pc in enumerate(violins["bodies"]):
        pc.set_edgecolor(lcol)
        pc.set_linewidth(0.5)
        if showbox:
            pc.set_alpha(1)
        pc.set_zorder(0.5)
        if i == 0:
            fc = pc.get_facecolor()
        else:
            fc
        pc.set_facecolor(fc)
    return ax
