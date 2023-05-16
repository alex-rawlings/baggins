import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from ..env_config import _cmlogger


__all__ = ["GradientLinePlot", "GradientScatterPlot"]


_logger = _cmlogger.copy(__file__)



class _GradientPlot:
    """
    
    """
    def __init__(self, ax, cmap="cividis", plot_kwargs={}):
        """
        Class to create pyplot plots with a colour gradient. The colour 
        gradient is consistent between all lines/points in the figure. This is 
        done by storing the data first, and then only plotting the data when 
        explicitly called.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axis to plot to
        cmap : str, optional
            pyplot colour map name, by default "cividis"
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
        try:
            dat_len = [len(v) for v in (x,y,c)]
            assert np.allclose(np.diff(dat_len), (0,0))
        except AssertionError:
            _logger.logger.exception(f"Input data must all be of the same length! Currently has lengths: {dat_len}", exc_info=True)
            raise
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_c.append(c)
        self.all_label.append(label)
        self.all_marker.append(marker)
        self.all_pks.append(plot_kwargs)
        self.data_count += 1


    def _set_colours(self, log=False, vmin=None, vmax=None):
        """
        Set the colours of the plot, should not be called directly

        Parameters
        ----------
        log : bool, optional
            colours in logscale?, by default False
        """
        if vmin is None:
            vmin = min([np.nanmin(ci) for ci in self.all_c])
        if vmax is None:
            vmax = max([np.nanmax(ci) for ci in self.all_c])
        if log:
            if vmin < 0:
                _logger.logger.warning("Log scale normalisation cannot handle negative values! Using a linear scale")
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
        """
        ax = self.ax if ax is None else ax
        plt.colorbar(plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), ax=ax, **kwargs)


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
            _logger.logger.exception(f"No data to plot!", exc_info=True)
            raise




class GradientLinePlot(_GradientPlot):
    """
    Apply the _GradientPlot class for pyplot line plots
    """
    def __init__(self, ax, cmap="viridis", plot_kwargs={}):
        super().__init__(ax, cmap=cmap, plot_kwargs=plot_kwargs)


    def plot(self, logcolour=False, ax=None, vmin=None, vmax=None):
        """
        Plot the data, ensuring a consistent colour scheme.

        Parameters
        ----------
        logcolour : bool, optional
            colours in log scale?, by default False
        """
        self._data_check()
        self._set_colours(log=logcolour, vmin=vmin, vmax=vmax)
        ax = self.ax if ax is None else ax
        for xi, yi, ci, labeli, markeri, pki in zip(self.all_x, self.all_y, self.all_c, self.all_label, self.all_marker, self.all_pks):
            if markeri is not None:
                ax.scatter(xi[-1], yi[-1], color=self.cmap(self.norm(ci[-1])), marker=markeri, label=labeli, zorder=10*self.data_count, ec="k", lw=0.5)
            for xs, ys, cs in zip(zip(xi[:-1], xi[1:]), zip(yi[:-1], yi[1:]), ci[:-1]):
                ax.plot(xs, ys, color=self.cmap(self.norm(cs)), **pki)




class GradientScatterPlot(_GradientPlot):
    """
    Apply the _GradientPlot class for pyplot scatter plots
    """
    def __init__(self, ax, x, y, c, label=None, cmap="viridis", marker="o", plot_kwargs={}):
        super().__init__(ax, x, y, c, label=label, cmap=cmap, marker=marker, plot_kwargs=plot_kwargs)


    def plot(self, logcolour=False, ax=None, vmin=None, vmax=None):
        """
        Plot the data, ensuring a consistent colour scheme.

        Parameters
        ----------
        logcolour : bool, optional
            colours in log scale?, by default False
        """
        self._data_check()
        self._set_colours(log=logcolour, vmin=vmin, vmax=vmax)
        ax = self.ax if ax is None else ax
        for xi, yi, ci, labeli, markeri, pki in zip(self.all_x, self.all_y, self.all_c, self.all_label, self.all_marker, self.all_pks):
            for i, (xs, ys, cs) in enumerate(zip(zip(xi[:-1], xi[1:]), zip(yi[:-1], yi[1:]), ci[:-1])):
                ax.scatter(xs, ys, color=self.cmap(self.norm(cs)), marker=markeri, label=(labeli if i==0 else ""), ec="k", lw=0.5, **pki)