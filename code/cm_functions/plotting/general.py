from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import colors
import itertools
from ..env_config import _logger


__all__ = ["draw_sizebar", "create_normed_colours", "mplColours", "mplLines", "mplChars", "shade_bool_regions"]


def draw_sizebar(ax, length, units, location="lower right", pad=0.1, borderpad=0.5, sep=5, frameon=False, unitconvert="base", remove_ticks=True):
    """
    Draw a horizontal scale bar using the mpl toolkit

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        axis to add the bar to
    length : float
        length of scale bar in data units
    units : str
        unit name
    location : str, optional
        where to place bar (standard pyplot location string), by default "lower 
        right"
    pad : float, optional
        padding around label, by default 0.1
    borderpad : float, optional
        padding around border, by default 0.5
    sep : float, optional
        separation between label and scale bar, by default 5
    frameon : bool, optional
        draw box around scale bar?, by default False
    unitconvert : str, optional
        convert units of scalebar, by default base (no conversion)
    remove_ticks : bool, optional
        remove tick labels on axis?, by default True
    """
    factors = {"mill2base":1e-3, "cent2base":1e-2, "base":1, "kilo2base":1e3, "mega2base":1e6}
    label = f"{length*factors[unitconvert]} {units}"
    asb = AnchoredSizeBar(ax.transData, length, label, loc=location, pad=pad, borderpad=borderpad, sep=sep, frameon=frameon)
    ax.add_artist(asb)
    if remove_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def create_normed_colours(vmin, vmax, cmap="viridis", normalisation="Normalize"):
    """
    Convenience wrapper for creating colour normalisation and colourbar 
    requirements for pyplot.plot()

    Parameters
    ----------
    vmin : float
        minimum value of colour variable
    vmax : float
        maximum value of colour variable
    cmap : str, optional
        pyplot colour map name, by default "viridis"
    normalisation: str, optional
        matplotlib.color attribute for normalisation

    Returns
    -------
    mapcols : function
        takes an argument in the range [vmin, vmax] and returns the scaled 
        colour
    sm matplotlib.cm.ScalarMappable
        object that is required for creating a colour bar
    """
    try:
        cmapv = getattr(plt.cm, cmap)
    except AttributeError:
        _logger.logger.warning(f"{cmap} does not exist. Using default colormap: viridis")
        cmapv = plt.cm.viridis
    try:
        _norm = getattr(colors, normalisation)
    except AttributeError:
        _logger.logger.warning(f"Normalisation {normalisation} is not valid. Using default (Linear).")
        _norm = colors.Normalize()
    norm = _norm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmapv)
    mapcols = lambda x: cmapv(norm(x))
    return mapcols, sm


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
    : collections.OrderedDict
        linestyles that can be accessed by keyword or .items() notation
    """
    return OrderedDict([
            ("solid",                    (0, ())),
            ("dotted",                   (0, (1,1))),
            ("dashed",                   (0, (5,regular))),
            ("dashdotted",               (0, (3,regular,1,regular))),
            ("dashdotdotted",            (0, (3,regular,1,regular,1,regular))),

            ("densely dashed",           (0, (5,dense))),
            ("densely dashdotted",       (0, (3,dense,1,dense))),
            ("densely dashdotdotted",    (0, (3,dense,1,dense,1,dense))),

            ("loosely dotted",           (0, (1,loose))),
            ("loosely dashed",           (0, (5,loose))),
            ("loosely dashdotted",       (0, (3,loose,1,loose))),
            ("loosely dashdotdotted",    (0,(3,loose,1,loose,1,loose))),
    ])


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
    ax : matplotlib.axes._subplots.AxesSubplot
        axis to plot to
    xdata : np.ndarray
        x data values
    mask : np.ndarray bool
        mask for xdata, will shade the true regions
    kwargs :
        keyword arguments for pyplot.axvspan()
    """
    #get the first and last index of the True "blocks"
    regions = [(group[0], group[-1]) for group in (list(group) for key, group in itertools.groupby(range(len(mask)), key=mask.__getitem__) if key)]
    for region in regions:
        ax.axvspan(xdata[region[0]], xdata[region[1]], **kwargs)

