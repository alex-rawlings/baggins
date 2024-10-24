from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from env_config import _cmlogger


__all__ = ["GradientLinePlot", "GradientScatterPlot"]


_logger = _cmlogger.getChild(__name__)


class _GradientPlot(ABC):
    def __init__(self, ax, cmap="cividis"):
        """
        Class to create pyplot plots with a colour gradient. The colour
        gradient is consistent between all lines/points in the figure. This is
        done by storing the data first, and then only plotting the data when
        explicitly called.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axis to plot to
        cmap : str, ListedColorMap, optional
            pyplot colour map name or instance, by default "cividis"
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
        if isinstance(cmap, colors.ListedColormap):
            self.cmap = cmap
        else:
            try:
                self.cmap = getattr(plt.cm, cmap)
            except AttributeError:
                try:
                    assert cmap in [
                        cm for cm in plt.colormaps() if cm not in dir(plt.cm)
                    ]
                    self.cmap = plt.colormaps[cmap]
                except AssertionError:
                    _logger.exception(
                        f"cmap `{cmap}` not present in matplotlib defaults, nor is registered as a custom map!",
                        exc_info=True,
                    )
                    raise
        self.all_marker = []
        self.norm = [0, 1]
        self.marker_kwargs = {"ec": "k", "lw": 0.5}

    def __len__(self):
        return self.data_count

    @property
    def max_colour(self):
        return max([max(c) for c in self.all_c])

    @property
    def min_colour(self):
        return min([min(c) for c in self.all_c])

    def add_data(self, x, y, c, label=None, marker="o"):
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
        label : str, optional
            label of plot, by default None
        marker : str, optional
            end marker, by default "o"
        """
        try:
            dat_len = [len(v) for v in (x, y, c)]
            assert np.allclose(np.diff(dat_len), (0, 0))
        except AssertionError:
            _logger.exception(
                f"Input data must all be of the same length! Currently has lengths: {dat_len}",
                exc_info=True,
            )
            raise
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_c.append(c)
        self.all_label.append(label)
        self.all_marker.append(marker)
        self.data_count += 1

    def _set_colours(self, log=False, vmin=None, vmax=None):
        """
        Set the colours of the plot, should not be called directly

        Parameters
        ----------
        log : bool, optional
            colours in logscale?, by default False
        vmin : float, optional
            enforce a minimum colour value, by default None
        vmax : float, optional
            enforce a maximum colour value, by default None
        """
        if vmin is None:
            vmin = min([np.nanmin(ci) for ci in self.all_c])
        if vmax is None:
            vmax = max([np.nanmax(ci) for ci in self.all_c])
        if log:
            if vmin < 0:
                _logger.warning(
                    "Log scale normalisation cannot handle negative values! Using a linear scale"
                )
                self.norm = colors.Normalize(vmin, vmax)
            else:
                self.norm = colors.LogNorm(vmin, vmax)
        else:
            self.norm = colors.Normalize(vmin, vmax)

    def add_cbar(self, ax=None, **kwargs):
        """
        Add a colour bar to the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axis to add to

        Returns
        -------
        : matplotlib.colorbar.Colorbar
            colorbar object
        """
        ax = self.ax if ax is None else ax
        return plt.colorbar(
            plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), ax=ax, **kwargs
        )

    def add_legend(self, ax, **kwargs):
        """
        Add a legend to the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axis to add to
        """
        ax = self.ax if ax is None else ax
        ax.legend(**kwargs)

    def _data_check(self):
        """
        Ensure the instance contains data to plot
        """
        try:
            assert self.data_count > 0
        except AssertionError:
            _logger.exception("No data to plot!", exc_info=True)
            raise


class GradientLinePlot(_GradientPlot):
    """
    Apply the _GradientPlot class for pyplot line plots
    """

    def __init__(self, ax, cmap="viridis"):
        super().__init__(ax, cmap=cmap)

    def _make_segments(self, i):
        points = np.array([self.all_x[i], self.all_y[i]]).T.reshape(-1, 1, 2)
        return np.concatenate([points[:-1], points[1:]], axis=1)

    def plot_single_series(
        self, i, logcolour=False, ax=None, vmin=None, vmax=None, marker_idx=-1, **kwargs
    ):
        """
        Plot the data for a single data series, ensuring colour scheme is
        consistent with all stored data

        Parameters
        ----------
        i : int
            index of data series to plot
        logcolour : bool, optional
            use logarithmic colour mapping, by default False
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        vmin : float, optional
            minimum colour value (overrides default value), by default None
        vmax : float, optional
            maximum colour value (ovverides default value), by default None
        marker_idx : int
            array index to place marker at
        kwargs :
            other keyword arguments to LineCollection()
        """
        self._data_check()
        self._set_colours(log=logcolour, vmin=vmin, vmax=vmax)
        ax = self.ax if ax is None else ax
        if self.all_marker[i] is not None:
            ax.scatter(
                self.all_x[i][marker_idx],
                self.all_y[i][marker_idx],
                color=self.cmap(self.norm(self.all_c[i][marker_idx])),
                marker=self.all_marker[i],
                label=self.all_label[i],
                zorder=10 * self.data_count,
                **self.marker_kwargs,
            )
        segments = self._make_segments(i)
        lc = LineCollection(
            segments, array=self.all_c[i], cmap=self.cmap, norm=self.norm, **kwargs
        )
        ax.add_collection(lc)
        return lc

    def plot(
        self, logcolour=False, ax=None, vmin=None, vmax=None, marker_idx=None, **kwargs
    ):
        """
        Plot the data for all data series, ensuring a consistent colour scheme.
        """
        if marker_idx is None:
            marker_idx = [-1 for _ in range(self.data_count)]
        for i in range(self.data_count):
            self.plot_single_series(
                i,
                logcolour=logcolour,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                marker_idx=marker_idx[i],
                **kwargs,
            )

    def draw_arrow_on_series(self, ax, i, idx0, direction="right", size=12):
        """
        Draw an arrow on a series.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            plotting axis
        i : int
            index of data series to plot
        idx0 : int
            index in data series i to start the arrow at
        direction : str, optional
            direction arrow points, by default "right"
        size : int, optional
            arrow size, by default 12
        """
        if direction == "left":
            idx1 = idx0 - 1
        else:
            idx1 = idx0 + 1
        ax.annotate(
            "",
            xytext=(self.all_x[i][idx0], self.all_y[i][idx0]),
            xy=(self.all_x[i][idx1], self.all_y[i][idx1]),
            arrowprops={
                "arrowstyle": "-|>",
                "fc": self.cmap(self.norm(self.all_c[i][idx0])),
                "ec": "w",
                "lw": 0.1,
            },
            size=size,
        )


class GradientScatterPlot(_GradientPlot):
    """
    Apply the _GradientPlot class for pyplot scatter plots
    """

    def __init__(self, ax, cmap="cividis"):
        super().__init__(ax, cmap=cmap)

    def _set_colours(self, log=False, vmin=None, vmax=None):
        """
        Set the colours for the plot, see similarly-called super() method

        Returns
        -------
        : callable
            function to map colours
        """
        super()._set_colours(log, vmin, vmax)
        return lambda x: self.cmap(self.norm(x))

    def plot_single_series(
        self, i, logcolour=False, ax=None, vmin=None, vmax=None, **kwargs
    ):
        """
        Plot the data for a single data series, ensuring colour scheme is
        consistent with all stored data

        Parameters
        ----------
        i : int
            index of data series to plot
        logcolour : bool, optional
            use logarithmic colour mapping, by default False
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        vmin : float, optional
            minimum colour value (overrides default value), by default None
        vmax : float, optional
            maximum colour value (ovverides default value), by default None
        kwargs :
            other keyword arguments to pyplot.scatter()
        """
        self._data_check()
        cmapper = self._set_colours(log=logcolour, vmin=vmin, vmax=vmax)
        ax = self.ax if ax is None else ax
        ax.scatter(self.all_x[i], self.all_y[i], c=cmapper(self.all_c[i]), **kwargs)

    def plot(self, logcolour=False, ax=None, vmin=None, vmax=None, **kwargs):
        """
        Plot the data, ensuring a consistent colour scheme.
        """
        for i in range(self.data_count):
            self.plot_single_series(
                i, logcolour=logcolour, ax=ax, vmin=vmin, vmax=vmax, **kwargs
            )
