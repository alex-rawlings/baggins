from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import itertools


__all__ = ["draw_sizebar", "mplColours", "mplLines", "mplChars", "shade_bool_regions"]


def draw_sizebar(ax, length, units, location="lower right", pad=0.1, borderpad=0.5, sep=5, frameon=False, unitconvert=None):
    """
    Draw a horizontal scale bar using the mpl toolkit

    Parameters
    ----------
    ax: axis to add the bar to
    length: length of scale bar in data units
    units: string stating unit name
    location: where to place bar (standard pyplot location string)
    pad: padding around label
    borderpad: padding around border
    sep: separation between label and scale bar
    frameon: draw box around scale bar
    unitconvert: convert units of scalebar

    Returns
    -------
        None (draw scale bar)
    """
    if unitconvert is None:
        label = str(length)+" "+units
    elif unitconvert == "kilo2base":
        label = str(length*1000)+" "+units
    else:
        #TODO other unit conversions
        raise NotImplementedError("Other unit conversions yet to be implemented")
    asb = AnchoredSizeBar(ax.transData, length, label, loc=location, pad=pad, borderpad=borderpad, sep=sep, frameon=frameon)
    ax.add_artist(asb)


def mplColours():
    """
    access the default matplotlib color palette by index
    maximum index:

    Parameters
    ----------
    None

    Returns
    -------
    the colour array, to be used in plt.plot(x,y,color=THIS)
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def mplLines(regular=5, loose=10, dense=1):
    """
    Create an ordered dictionary that allows for different linestyles. The
    default parameter values are taken from the matplotlib example.

    Parameters
    ----------
    regular: spacing between lines/points for "normal" appearance
    loose: spacing between lines/points for "loose" appearance
    dense: spacing between lines/points for "dense" appearance

    Returns
    -------
    OrderedDict, with linestyles that can be accessed by keyword or .items()
    notation
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
    Return a list of matplotlib plotting characters
    """
    return ["o", "s", "^", "D", "v", "*", "p", "h", "X", "P"]


def shade_bool_regions(ax, xdata, mask, **kwargs):
    """
    Shade regions of plot corresponding to the True regions of a mask

    Parameters
    ----------
    ax: matplotlib axis object (where the plotting will be done)
    xdata: array of x data values
    mask: mask for xdata, will shade the true regions
    **kwargs: keyword arguments for pyplot.axvspan()

    Returns
    -------
    None
    """
    #get the first and last index of the True "blocks"
    regions = [(group[0], group[-1]) for group in (list(group) for key, group in itertools.groupby(range(len(mask)), key=mask.__getitem__) if key)]
    for region in regions:
        ax.axvspan(xdata[region[0]], xdata[region[1]], **kwargs)


def draw_sizebar(ax, length, units, location='lower right', pad=0.1, borderpad=0.5, sep=5, frameon=False, unitconvert=None, remove_ticks=True):
    """
    Draw a horizontal scale bar using the mpl toolkit
    
    Parameters
    ----------
    ax: pyplot axis to add the bar to
    length: length of scale bar in data units
    units: string stating unit name
    location: where to place bar (standard pyplot location string)
    pad: padding around label
    borderpad: padding around border
    sep: separation between label and scale bar
    frameon: draw box around scale bar
    unitconvert: convert units of scalebar
    remove_ticks: remove tick labels on axis?
    """
    if unitconvert is None:
        label = str(length)+' '+units
    elif unitconvert == 'kilo2base':
        label = str(length*1000)+' '+units
    else:
        #TODO other unit conversions
        raise NotImplementedError('Other unit conversions yet to be implemented')
    asb = AnchoredSizeBar(ax.transData, length, label, loc=location, pad=pad, borderpad=borderpad, sep=sep, frameon=frameon)
    ax.add_artist(asb)
    if remove_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)