import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import scipy.interpolate


__all__ = ['draw_sizebar', 'mplColours']


def draw_sizebar(ax, length, units, location='lower right', pad=0.1, borderpad=0.5, sep=5, frameon=False, unitconvert=None):
    '''
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
    '''
    if unitconvert is None:
        label = str(length)+' '+units
    elif unitconvert == 'kilo2base':
        label = str(length*1000)+' '+units
    else:
        #TODO other unit conversions
        raise NotImplementedError('Other unit conversions yet to be implemented')
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
    return plt.rcParams['axes.prop_cycle'].by_key()['color']
