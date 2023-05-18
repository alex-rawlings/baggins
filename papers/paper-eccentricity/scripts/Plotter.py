import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
import matplotlib.patheffects as pe



class Plotter:
    def __init__(self) -> None:
        """
        Rudimentary wrapper to make plotting easier
        """
        self._get_col = None
        self._color_constructor()
        self._get_marker = lambda x: ["o", "s", "^", "D", "v", "*", "p", "h", "X", "P"][x%10]
        self._plot_linewidth = rcParams["lines.linewidth"]+2
        self._line_count = {}
        self._scatter_count = {}
        self._error_count = {}
        self._markerkwargs = {"ec":"k", "lw":0.5}


    def _color_constructor(self):
        """
        Construct the colour palette: a reordered, 10-element version of the 
        `twilight` colour map.
        """
        cols_f = lambda x: plt.cm.twilight(colors.Normalize(0,10)(x))
        _cols = [cols_f(i) for i in range(10)]
        cols_1 = _cols[:5]
        cols_2 = _cols[5:]
        cols = []
        for c1, c2 in zip(cols_1, cols_2):
            cols.append(c1)
            cols.append(c2)
        cols.extend(cols[:2])
        for i in range(2): cols.pop(0)
        self._get_col = lambda x: cols[x%10]


    def _get_ax(self, ax):
        """
        Convenience routine to get the axis to plot to

        Returns
        -------
        : matplotlib.axes.Axes
            plotting axis
        """
        return plt.gca() if ax is None else ax


    def _get_counts(self, ax, counter):
        if ax.__hash__() not in counter:
            counter[ax.__hash__()] = 0
        return counter[ax.__hash__()]


    def _update_counts(self, ax, counter):
        d = getattr(self, counter)
        d[ax.__hash__()] += 1
        setattr(self, counter, d)


    def print_all_counts(self):
        hl = set(self._line_count.keys())
        hs = set(self._scatter_count.keys())
        he = set(self._error_count.keys())
        h = hl.union(hs).union(he)
        for hh in h:
            print(f"Axes: {hh}")
            for c, s in zip((self._line_count, self._scatter_count, self._error_count), ("Line", "Scatter", "Errorbar")):
                try:
                    cc = c[hh]
                except KeyError:
                    cc = 0
                print(f"  > {s} count: {cc}")


    def plot(self, x, y, ax=None, **kwargs):
        """
        Wrapper around pyplot.plot() that implements the colour scheme and edge 
        effects

        Parameters
        ----------
        x : array-like
            x data to plot
        y : array-like
            y data to plot
        ax : matplotlib.axes.Axes, optional
            plotting axis, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes, optional
            plotting axis
        """
        ax = self._get_ax(ax)
        lc = self._get_counts(ax, self._line_count)
        ax.plot(
                x, y, 
                c=self._get_col(lc), 
                lw=self._plot_linewidth, 
                path_effects=[pe.Stroke(linewidth=self._plot_linewidth+1.5, foreground="k"), pe.Normal()], 
                **kwargs
            )
        self._update_counts(ax, "_line_count")
        return ax


    def scatter(self, x, y, ax=None, **kwargs):
        """
        Wrapper around pyplot.scatter() that implements the colour scheme and 
        edge effects

        Parameters
        ----------
        x : array-like
            x data to plot
        y : array-like
            y data to plot
        ax : matplotlib.axes.Axes, optional
            plotting axis, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes, optional
            plotting axis
        """
        ax = self._get_ax(ax)
        sc = self._get_counts(ax, self._scatter_count)
        ax.scatter(
                    x, y, 
                    fc=self._get_col(sc), 
                    marker=self._get_marker(sc), 
                    **self._markerkwargs, 
                    **kwargs
                )
        self._update_counts(ax, "_scatter_count")
        return ax


    def errorbar(self, x, y, ax=None, **kwargs):
        """
        Wrapper around pyplot.errorbar() that implements the colour scheme and 
        edge effects

        Parameters
        ----------
        x : array-like
            x data to plot
        y : array-like
            y data to plot
        ax : matplotlib.axes.Axes, optional
            plotting axis, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes, optional
            plotting axis
        """
        ax = self._get_ax(ax)
        ec = self._get_counts(ax, self._error_count)
        ax.errorbar(
                    x, y, 
                    c=self._get_col(ec), 
                    fmt=self._get_marker(ec),
                    ls="", 
                    elinewidth=self._plot_linewidth,
                    capsize=4,
                    mec=self._markerkwargs["ec"], 
                    mew=self._markerkwargs["lw"],
                    **kwargs
                    )
        self._update_counts(ax, "_error_count")
        return ax


