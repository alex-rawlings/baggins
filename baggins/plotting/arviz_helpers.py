import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import arviz as az
from xarray import Dataset
from baggins.env_config import _cmlogger

__all__ = ["plot_hdi", "get_all_axes_from_plot_collection"]

_logger = _cmlogger.getChild(__name__)


def get_all_axes_from_plot_collection(pc):
    """
    Get all axes from an arviz PlotCollection object as an array.

    Parameters
    ----------
    pc : arivz_plots.PlotCollection
        object to get axes from

    Returns
    -------
    : np.ndarray
        flattened array of plotting axes
    """
    plots = pc.viz["plot"].to_dataset()
    return np.concatenate([np.ravel(da.values) for da in plots.data_vars.values()])


def plot_hdi(
    x,
    y=None,
    hdi_prob=None,
    hdi_data=None,
    color="C1",
    circular=False,
    smooth=True,
    smooth_kwargs=None,
    figsize=None,
    fill_kwargs=None,
    plot_kwargs=None,
    hdi_kwargs=None,
    ax=None,
):
    """
    Plot HDI intervals for regression data. This function is taekn from arviz v 0.2.0

    Parameters
    ----------
    x : array-like
        Values to plot.
    y : array-like, optional
        Values from which to compute the HDI. Assumed shape ``(chain, draw, \*shape)``.
        Only optional if ``hdi_data`` is present.
    hdi_data : array_like, optional
        Precomputed HDI values to use. Assumed shape is ``(*x.shape, 2)``.
    hdi_prob : float, optional
        Probability for the highest density interval. Defaults to ``stats.ci_prob`` rcParam.
        See :ref:`this section <common_ hdi_prob>` for usage examples.
    color : str, default "C1"
        Color used for the limits of the HDI and fill. Should be a valid matplotlib color.
    circular : bool, default False
        Whether to compute the HDI taking into account ``x`` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
    smooth : boolean, default True
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
    smooth_kwargs : dict, optional
        Additional keywords modifying the Savitzky-Golay filter. See
        :func:`scipy:scipy.signal.savgol_filter` for details.
    figsize : (float, float), optional
        Figure size. If ``None``, it will be defined automatically.
    fill_kwargs : dict, optional
        Keywords passed to :meth:`mpl:matplotlib.axes.Axes.fill_between`
        (use ``fill_kwargs={'alpha': 0}`` to disable fill) or to
        :meth:`bokeh.plotting.Figure.patch`.
    plot_kwargs : dict, optional
        HDI limits keyword arguments, passed to :meth:`mpl:matplotlib.axes.Axes.plot` or
        :meth:`bokeh.plotting.Figure.patch`.
    hdi_kwargs : dict, optional
        Keyword arguments passed to :func:`~arviz.hdi`. Ignored if ``hdi_data`` is present.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes or bokeh figures.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        plotting axes
    """
    if hdi_kwargs is None:
        hdi_kwargs = {}

    x = np.asarray(x)
    x_shape = x.shape

    if y is None and hdi_data is None:
        raise ValueError("One of {y, hdi_data} is required")
    if hdi_data is not None and y is not None:
        _logger.warning("Both y and hdi_data arguments present, ignoring y")
    elif hdi_data is not None:
        hdi_prob = (
            hdi_data.hdi.attrs.get("hdi_prob", np.nan)
            if hasattr(hdi_data, "hdi")
            else np.nan
        )
        if isinstance(hdi_data, Dataset):
            data_vars = list(hdi_data.data_vars)
            if len(data_vars) != 1:
                raise ValueError(
                    "Found several variables in hdi_data. Only single variable Datasets are "
                    "supported."
                )
            hdi_data = hdi_data[data_vars[0]]
    else:
        y = np.asarray(y)
        if hdi_prob is None:
            hdi_prob = az.rcParams["stats.ci_prob"]
        elif not 1 >= hdi_prob > 0:
            raise ValueError("The value of hdi_prob should be in the interval (0, 1]")
        hdi_data = az.hdi(y, prob=hdi_prob, circular=circular, **hdi_kwargs)

    hdi_shape = hdi_data.shape
    if hdi_shape[:-1] != x_shape:
        msg = (
            "Dimension mismatch for x: {} and hdi: {}. Check the dimensions of y and"
            "hdi_kwargs to make sure they are compatible"
        )
        raise TypeError(msg.format(x_shape, hdi_shape))

    if smooth:
        if isinstance(x[0], np.datetime64):
            raise TypeError(
                "Cannot deal with x as type datetime. Recommend setting smooth=False."
            )

        if smooth_kwargs is None:
            smooth_kwargs = {}
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        x_data = np.linspace(x.min(), x.max(), 200)
        x_data[0] = (x_data[0] + x_data[1]) / 2
        hdi_interp = griddata(x, hdi_data, x_data)
        y_data = savgol_filter(hdi_interp, axis=0, **smooth_kwargs)
    else:
        idx = np.argsort(x)
        x_data = x[idx]
        y_data = hdi_data[idx]

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("color", color, None)
    plot_kwargs.setdefault("alpha", 0, None)
    if fill_kwargs is None:
        fill_kwargs = {}
    fill_kwargs.setdefault("color", color, None)
    fill_kwargs.setdefault("alpha", 0.5, None)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)

    return ax
