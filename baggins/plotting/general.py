import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import colors
import itertools
from baggins.env_config import _cmlogger


__all__ = [
    "draw_sizebar",
    "create_normed_colours",
    "create_offcentre_diverging",
    "mplColours",
    "mplLines",
    "mplChars",
    "shade_bool_regions",
    "create_odd_number_subplots",
    "nice_log10_scale",
    "arrow_on_line",
    "add_log_guiding_gradients",
    "extract_contours_from_plot",
]

_logger = _cmlogger.getChild(__name__)


def draw_sizebar(
    ax,
    length,
    units,
    unitconvert="base",
    remove_ticks=True,
    fmt=".1f",
    **kwargs,
):
    """
    Draw a horizontal scale bar using the mpl toolkit

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axis to add the bar to
    length : float
        length of scale bar in data units
    units : str
        unit name
    unitconvert : str, optional
        convert units of scalebar, by default base (no conversion)
    remove_ticks : bool, optional
        remove tick labels on axis?, by default True
    fmt : str, optional
        formatter for numeric part of label, by default ".1f"
    kwargs :
        other keyword arguments for AnchoredSizeBar()

    Returns
    -------
    asb : matplotlib.Artist
        anchor bar artist
    """
    factors = {
        "mill2base": 1e-3,
        "cent2base": 1e-2,
        "base": 1,
        "kilo2base": 1e3,
        "mega2base": 1e6,
    }
    label = f"$\mathrm{{{length*factors[unitconvert]:{fmt}}\,{units}}}$"
    if "loc" not in kwargs:
        loc = kwargs.pop("location", "lower right")
    kwargs.setdefault("loc", loc)
    kwargs.setdefault("pad", 0.1)
    kwargs.setdefault("borderpad", 0.5)
    kwargs.setdefault("frameon", False)
    textsize = kwargs.pop("textsize", None)
    kwargs.setdefault("fontproperties", {"size": textsize})
    asb = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        **kwargs,
    )
    ax.add_artist(asb)
    if remove_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    return asb


def create_normed_colours(
    vmin,
    vmax,
    cmap="cividis",
    norm="Normalize",
    norm_kwargs={},
    trunc=(None, None),
    bad=None,
):
    """
    Convenience wrapper for creating colour normalisation and colourbar
    requirements for pyplot.plot()
    # TODO this doesn't work with colors.CenteredNorm() due to different
    # argument names

    Parameters
    ----------
    vmin : float
        minimum value of colour variable
    vmax : float
        maximum value of colour variable
    cmap : str, optional
        pyplot colour map name, by default "cividis"
    norm: str, optional
        matplotlib.color attribute for normalisation
    norm_kwargs: dict, optional
        additional keyword arguments for normalisation initialisation
    trunc: tuple, optional
        values to truncate colour map to, by default (None, None)

    Returns
    -------
    mapcols : function
        takes an argument in the range [vmin, vmax] and returns the scaled
        colour
    sm : matplotlib.cm.ScalarMappable
        object that is required for creating a colour bar
    """
    try:
        cmapv = plt.get_cmap(cmap)
    except ValueError:
        _logger.warning(f"{cmap} does not exist. Using default colormap: cividis")
        cmapv = plt.get_cmap("cividis")
    if bad is not None:
        cmapv.set_bad(color=bad)
    try:
        _norm = getattr(colors, norm)
    except AttributeError:
        _logger.warning(f"Normalisation {norm} is not valid. Using default (Linear).")
        _norm = colors.Normalize()
    norm = _norm(vmin=vmin, vmax=vmax, **norm_kwargs)
    mapcols = lambda x: cmapv(norm(x))
    # we now have a colormap that maps (vmin, vmax) -> (0,1)
    # adjust if we want to truncate it
    if any([tt is not None for tt in trunc]):
        tmin = vmin if trunc[0] is None else trunc[0]
        tmax = vmax if trunc[1] is None else trunc[1]
        col_arr = mapcols(np.linspace(tmin, tmax, 256))
        cmapv = colors.LinearSegmentedColormap.from_list("trunc", col_arr)
        mapcols = lambda x: cmapv(norm(x))
        norm = _norm(vmin=tmin, vmax=tmax, **norm_kwargs)
        _logger.debug(
            f"Truncating colormap from ({vmin:.2f},{vmax:.2f}) --> ({tmin:.2f},{tmax:.2f})"
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmapv)
    return mapcols, sm


def create_offcentre_diverging(vmin, vmax, vcentre=0, cmap="seismic"):
    """
    Create a diverging colourmap centred about some value. The colours are mapped to the extent that has the larger magnitude, and then truncated to just those values given by the desired colour limits.

    Parameters
    ----------
    vmin : float
        minimum value of colour variable
    vmax : float
        maximum value of colour variable
    vcentre : float, optional
        value of central colour variable, by default 0
    cmap : str or matplotlib.colors.ListedColormap, optional
        colour map to use, by default "seismic"

    Returns
    -------
    : function
        takes an argument in the range [vmin, vmax] and returns the scaled
        colour
    sm : matplotlib.cm.ScalarMappable
        object that is required for creating a colour bar
    """
    # first create the diverging colour scheme
    if isinstance(cmap, str):
        _cmapv = plt.get_cmap(cmap)
    else:
        _cmapv = cmap
    _norm = colors.CenteredNorm(
        vcenter=vcentre, halfrange=max(np.abs(vmax), np.abs(vmin))
    )
    # now get the values we wish to restrict to
    u = np.linspace(vmin, vmax, 256)
    col_list = _cmapv(_norm(u))
    cmapv = colors.LinearSegmentedColormap.from_list("custom", col_list)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm, cmap=cmapv)
    return lambda x: cmapv(norm(x)), sm


def mplColours():
    """
    access the default matplotlib color palette by index

    Returns
    -------
    : list
        colour array, to be used in plt.plot(x,y,color=THIS[index])
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def mplLines(regular=5, loose=10, dense=1):
    """
    Create an ordered dictionary that allows for different linestyles. The
    default parameter values are taken from the matplotlib example.

    Parameters
    ----------
    regular : int, optional
        spacing between lines/points for "normal" appearance, by default 5
    loose : int, optional
        spacing between lines/points for "loose" appearance, by default 10
    dense : int, optional
        spacing between lines/points for "dense" appearance, by default 1

    Returns
    -------
    : list
        linestyles
    """
    d = dict(
        [
            ("solid", (0, ())),
            ("dotted", (0, (1, 1))),
            ("dashed", (0, (5, regular))),
            ("dashdotted", (0, (3, regular, 1, regular))),
            ("dashdotdotted", (0, (3, regular, 1, regular, 1, regular))),
            ("densely-dashed", (0, (5, dense))),
            ("densely-dashdotted", (0, (3, dense, 1, dense))),
            ("densely-dashdotdotted", (0, (3, dense, 1, dense, 1, dense))),
            ("loosely-dotted", (0, (1, loose))),
            ("loosely-dashed", (0, (5, loose))),
            ("loosely-dashdotted", (0, (3, loose, 1, loose))),
            ("loosely-dashdotdotted", (0, (3, loose, 1, loose, 1, loose))),
        ]
    )
    return list(d.values())


def mplChars():
    """
    list of matplotlib plotting characters

    Returns
    -------
    : list
        plotting characters
    """
    return ["o", "s", "^", "D", "v", "*", "p", "h", "X", "P"]


def shade_bool_regions(ax, xdata, mask, **kwargs):
    """
    Shade regions of plot corresponding to the True regions of a mask

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axis to plot to
    xdata : np.ndarray
        x data values
    mask : np.ndarray bool
        mask for xdata, will shade the true regions
    kwargs :
        keyword arguments for pyplot.axvspan()
    """
    # get the first and last index of the True "blocks"
    regions = [
        (group[0], group[-1])
        for group in (
            list(group)
            for key, group in itertools.groupby(range(len(mask)), key=mask.__getitem__)
            if key
        )
    ]
    for region in regions:
        ax.axvspan(xdata[region[0]], xdata[region[1]], **kwargs)


def create_odd_number_subplots(nrow, ncol, fkwargs={}, gskwargs={}):
    """
    Create a set of subplots where the final row has an odd number of panels
    (1 less than the previous rows).

    Parameters
    ----------
    nrow : int
        number of rows
    ncol : int
        number of columns for all but the last row, which will have 1-ncol
        columns
    fkwargs : dict, optional
        kwargs dict to be parsed to plt.figure(), by default {}
    gskwargs : dict, optional
        kwargs dict to be parsed to GridSpec(), by default {}

    Returns
    -------
    _type_
        _description_
    """
    fig = plt.figure(**fkwargs)
    ax = []
    if "top" not in gskwargs:
        gskwargs["top"] = 0.95
    gs = GridSpec(nrow, 2 * ncol, figure=fig, **gskwargs)
    # "normal" rows
    for j in range(nrow - 1):
        for i in range(ncol):
            ax.append(fig.add_subplot(gs[j, 2 * i : 2 * (i + 1)]))
    # final row, with odd number of panels
    for i in range(ncol - 1):
        ax.append(fig.add_subplot(gs[-1, 2 * i + 1 : 2 * (i + 1) + 1]))
    return fig, np.array(ax)


def nice_log10_scale(ax, axis="y"):
    """
    Ensure nice log scaling on an axis

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axis to plot to
    axis : str, optional
        edit done to x or y axis, by default "y"
    """
    if "y" in axis:
        ylims = ax.get_ylim()
        ax.set_ylim(
            10 ** np.floor(np.log10(ylims[0])), 10 ** np.ceil(np.log10(ylims[1]))
        )
    if "x" in axis:
        xlims = ax.get_xlim()
        ax.set_xlim(
            10 ** np.floor(np.log10(xlims[0])), 10 ** np.ceil(np.log10(xlims[1]))
        )


def arrow_on_line(ln, xpos=None, direction="right", size=15, arrowprops={}):
    """
    Add an arrow to a curve

    Parameters
    ----------
    ln : matplotib.Line2D
        line object to add arrow to
    xpos : float, optional
        x-coordinate to draw line on, by default None
    direction : str, optional
        direction arrow points, by default "right"
    size : int, optional
        size of arrow, by default 15
    arrowprops : dict, optional
        arrow style kwargs, by default {}
    """
    if arrowprops:
        arrowprops = {"arrowstyle": "->", "color": ln.get_color()}
    xdata = ln.get_xdata()
    ydata = ln.get_ydata()

    if xpos is None:
        xpos = xdata.mean()
    # roughly get start index
    idx0 = np.argim(np.abs(xdata - xpos))
    if direction == "left":
        idx1 = idx0 - 1
    else:
        idx1 = idx0 + 1
    ln.axes.annotate(
        "",
        xytext=(xdata[idx0], ydata[idx0]),
        xy=(xdata[idx1], ydata[idx1]),
        arrowprops=arrowprops,
        size=size,
    )


def add_log_guiding_gradients(ax, x0, x1, y1, b, offset=0, fmt="+.0f", **kwargs):
    """
    Add guiding gradient lines to a log-log plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axis to plot to_
    x0 : float
        lower x limit of line
    x1 : float
        upper x limit of line
    y1 : float
        y-intersection point of lines
    b : list
        gradients to plot
    offset : float, optional
        text offset, by default 0
    fmt : str, optional
        label formatting string, by default "+.0f"
    """
    xlines = np.array([x0, x1])
    kwargs.setdefault("ls", "-")
    kwargs.setdefault("c", "k")
    kwargs.setdefault("lw", 1)
    for _b in b:
        ax.plot(xlines, y1 * (xlines / x1) ** _b, **kwargs)
        ax.text(
            x0 + offset,
            y1 * ((x0 + offset) / x1) ** _b,
            rf"$\propto {_b:{fmt}}$",
            ha="right",
            va="center",
        )


def extract_contours_from_plot(p):
    """
    Extract the x-y coordinates of contours from a contour plot.

    Parameters
    ----------
    p : matplotlib.contour.QuadContourSet
        output from a pyplot.contour() call

    Returns
    -------
    x : list
        list of x coordinates for contours
    y : list
        list of y coordinates for contours
    """
    x = []
    y = []
    for i, collection in enumerate(p.collections):
        for path in collection.get_paths():
            v = path.vertices
            x.append(v[:, 0])
            y.append(v[:, 1])
    return x, y
