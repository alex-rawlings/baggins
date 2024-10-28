import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from baggins.env_config import _cmlogger, this_dir
from baggins.mathematics import stat_interval, vertical_RMSE
from baggins.utils import create_error_col

__all__ = ["LiteratureTables"]

_logger = _cmlogger.getChild(__name__)


class LiteratureTables:
    def __init__(self) -> None:
        """
        Convenience class for loading data included with the baggins
        package. Cleaning/reformatting of the data is wrapped up into load()
        functions.

        Parameters
        ----------
        table_name : str
            name of the data table to load

        Raises
        ------
        ValueError
            if an invalid table name is given
        """
        self.table = None
        self._literature_dir = os.path.join(this_dir, "literature/literature_data")
        self.name = None

    @classmethod
    def load_sahu_2020_data(cls):
        """
        Load the BH-Bulge relation of Sahu 2020

        Returns
        -------
        : LiteratureTables
            table instance
        """
        C = cls()
        C.name = "Sahu+20"
        data = pd.read_table(
            os.path.join(C._literature_dir, "sahu_20.txt"), sep=",", header=0
        )
        data["Re_maj_kpc"] = data.loc[:, "Re_maj"] * data.loc[:, "scale"]
        data["logRe_maj_kpc"] = np.log10(data["Re_maj_kpc"])
        # restrict to only ETGs (exclude also S0)
        # data = data.loc[np.logical_or(data.loc[:,"Type"]=="E", data.loc[:,"Type"]=="ES"), :]
        cored_galaxies = np.zeros(data.shape[0], dtype="bool")
        for ind, gal in enumerate(data.loc[:, "Galaxy"]):
            if gal[-1] == "a":
                cored_galaxies[ind] = 1
        data.insert(2, "Cored", cored_galaxies)
        create_error_col(data, "logM*_sph")
        create_error_col(data, "logMbh")
        C.table = data
        return C

    @classmethod
    def load_sdss_mass_data(cls):
        """
        Load stellar masses from SDSS DR7

        Returns
        -------
        : LiteratureTables
            table instance
        """
        C = cls()
        C.name = "SDSS"
        C.table = pd.read_table(os.path.join(C._literature_dir, "sdss_z.csv"), sep=",")
        return C

    @classmethod
    def load_jin_2020_data(cls):
        """
        Load inner DM fraction from orbit modelling by Jin 2020

        Returns
        -------
        pd.DataFrame
            table
        """
        C = cls()
        C.name = "Jin+20"
        C.table = pd.read_fwf(
            os.path.join(C._literature_dir, "jin_2020.dat"),
            comment="#",
            names=[
                "MaNGAID",
                "log(M*/Msun)",
                "Re(kpc)",
                "f_DM",
                "p_e",
                "q_e",
                "T_e",
                "f_cold",
                "f_warm",
                "f_hot",
                "f_CR",
                "f_prolong",
                "f_CRlong",
                "f_box",
                "f_SR",
            ],
        )
        return C

    @classmethod
    def load_vdBosch_2016_data(cls):
        """
        Load data for the BH mass - stellar velocity dispersion relation of
        van den Bosch 2016

        Returns
        -------
        : LiteratureTables
            table instance
        """
        C = cls()
        C.name = "van den Bosch 16"
        data = pd.read_table(
            os.path.join(C._literature_dir, "bosch_16.txt"),
            sep=";",
            header=0,
            skiprows=[1],
        )
        # clean data
        data.loc[
            data.loc[:, "e_logBHMass"] == data.loc[:, "logBHMass"], "e_logBHMass"
        ] = np.nan
        data.loc[data.loc[:, "logBHMass"] < 1, "logBHMass"] = np.nan
        C.table = data
        return C

    @classmethod
    def load_thomas_2016_data(cls):
        """
        Load core radius, BH mass data from Thomas et al. 2016

        Returns
        -------
        : LiteratureTables
            table instance
        """
        C = cls()
        C.name = r"$\mathrm{Thomas\; et\; al.\; 2016}$"
        data = pd.read_csv(
            os.path.join(C._literature_dir, "thomas_16.csv"),
            header=0,
            skipinitialspace=True,
        )
        C.table = data
        C.table["log_BH_mass"] = np.log10(C.table.loc[:, "BH_mass"])
        C.table["log_core_radius_kpc"] = np.log10(C.table.loc[:, "core_radius_kpc"])
        return C

    @classmethod
    def load_dullo_2019_data(cls):
        """
        Load core radius, BH mass data from Thomas et al. 2016

        Returns
        -------
        : LiteratureTables
            table instance
        """
        C = cls()
        C.name = "Dullo 2019"
        data = pd.read_csv(
            os.path.join(C._literature_dir, "dullo_19.csv"),
            header=0,
            skipinitialspace=True,
        )
        C.table = data
        C.table["core_radius_kpc"] = C.table.loc[:, "core_radius_pc"] / 1e3
        C.table["log_BH_mass"] = np.log10(C.table.loc[:, "BH_mass"])
        C.table["log_core_radius_kpc"] = np.log10(C.table.loc[:, "core_radius_kpc"])
        return C

    def hist(self, var, ax=None, hist_kwargs={}):
        """
        Histogram table data

        Parameters
        ----------
        var : str
            column to histogram
        ax : matplotlib.axes.Axes, optional
            axes to plot to, by default None
        hist_kwargs : dict, optional
            plotting parameters parsed to plt.hist(), by default {}

        Returns
        -------
        matplotlib.axes.Axes
            plotting axis
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        default_hist_kwargs = {
            "bins": 20,
            "density": True,
            "facecolor": "tab:blue",
            "alpha": 0.5,
            "linewidth": 0.8,
            "edgecolor": "k",
        }
        for k, v in hist_kwargs.items():
            default_hist_kwargs[k] = v
        ax.hist(self.table.loc[:, var], label=self.name, **default_hist_kwargs)
        return ax

    def add_qauntile_to_plot(self, q, var, ax, xaxis=True, lkwargs={}):
        """
        Add a line to a plot representing a quantile

        Parameters
        ----------
        q : float
            quantile to plot
        var : str
            column to histogram
        ax : matplotlib.axes.Axes
            axes to plot to
        xaxis : bool, optional
            plot on the x-axis?, by default True
        lkwargs : dict, optional
            plotting parameters passed to axvline() or axhline(), by default {}
        """
        default_lkwargs = {"c": "tab:red"}
        label = f"{q} Quantile" if q != 0.5 else "Median"
        for k, v in lkwargs.items():
            default_lkwargs[k] = v
        if xaxis:
            ax.axvline(
                np.nanquantile(self.table.loc[:, var], q),
                label=label,
                **default_lkwargs,
            )
        else:
            ax.axhline(
                np.nanquantile(self.table.loc[:, var], q),
                label=label,
                **default_lkwargs,
            )

    def scatter(
        self, x, y, xerr=None, yerr=None, ax=None, scatter_kwargs={}, mask=None
    ):
        """
        Create a scatter plot of two columns of the table.

        Parameters
        ----------
        x : str
            column to plot on the x axis
        y : str
            column to plot on the y axis
        xerr : str, list, tuple, optional
            column for x-errors, or list/tuple of strings for non-symmetric
            errors, by default None
        yerr : str, list, tuple, optional
            column for y-errors, or list/tuple of strings for non-symmetric
            errors, by default None
        ax : matplotlib.axes.Axes, optional
            axes to plot to, by default None
        scatter_kwargs : dict, optional
            scatter parameters parsed to scatter() or errorbar(), by default {}
        mask : pd.Series, array-like, optional
            boolean mask to apply to all columns, by default None

        Returns
        -------
        matplotlib.axes.Axes
            plotting axis
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        default_scatter_kwargs = {"marker": ".", "alpha": 0.4, "c": "k", "zorder": 2}
        if xerr is not None or yerr is not None:
            default_scatter_kwargs["elinewidth"] = 1
        default_scatter_kwargs.update(scatter_kwargs)

        if xerr is None and yerr is None:
            # no error bars
            if mask is None:
                ax.scatter(
                    self.table.loc[:, x],
                    self.table.loc[:, y],
                    label=self.name,
                    **default_scatter_kwargs,
                )
            else:
                ax.scatter(
                    self.table.loc[mask, x],
                    self.table.loc[mask, y],
                    label=self.name,
                    **default_scatter_kwargs,
                )
        else:
            default_scatter_kwargs["fmt"] = default_scatter_kwargs["marker"]
            del default_scatter_kwargs["marker"]
            # error bars
            if mask is None:
                if xerr is not None:
                    if isinstance(xerr, (list, tuple)):
                        xerr = [self.table.loc[:, xerr[0]], self.table.loc[:, xerr[1]]]
                    else:
                        xerr = self.table.loc[:, xerr]
                if yerr is not None:
                    if isinstance(yerr, (list, tuple)):
                        yerr = [self.table.loc[:, yerr[0]], self.table.loc[:, yerr[1]]]
                    else:
                        yerr = self.table.loc[:, yerr]
                ax.errorbar(
                    self.table.loc[:, x],
                    self.table.loc[:, y],
                    xerr=xerr,
                    yerr=yerr,
                    ls="",
                    label=self.name,
                    **default_scatter_kwargs,
                )
            else:
                if xerr is not None:
                    xerr = self.table.loc[mask, xerr]
                if yerr is not None:
                    yerr = self.table.loc[mask, yerr]
                ax.errorbar(
                    self.table.loc[mask, x],
                    self.table.loc[mask, y],
                    xerr=xerr,
                    yerr=yerr,
                    ls="",
                    label=self.name,
                    **default_scatter_kwargs,
                )
        return ax

    def plot_lin_regress(
        self,
        x,
        y,
        itype="conf",
        conf_lev=0.68,
        xhat_method=np.linspace,
        ax=None,
        fit_in_log=False,
        mask=None,
        scatter_kwargs={},
        fit_coeffs={},
    ):
        """
        Convenience method to plot 2D data from a table as regression

        Parameters
        ----------
        x : str, array-like
            data key for independent variable, or data directly
        y : str, array-like
            data key for dependent variable, or data directly
        itype : str, optional
            type of statistic interval, by default "conf"
        conf_lev : float, optional
            level of statistic interval, by default 0.68
        xhat_method : callable, optional
            method to generate evenly spaced independent variables, by default
            np.linspace
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        fit_in_log : bool, optional
            fit regression in log space, by default False
        mask : array-like, optional
            mask to fit regression to subset of data, by default None
        scatter_kwargs : dict, optional
            scatter parameters parsed to scatter() or errorbar(), by default {}
        fit_coeffs : dict, optional
            linear regression fit coefficients (keys must be 'slope' and
            'intercept') to be used instead of fitting coefficients
            independently, by default {}

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axes
        """
        if mask is None:
            mask = np.ones(len(self.table), dtype=bool)
        if fit_in_log:
            _x = self.table.loc[mask, f"log_{x}"]
            _y = self.table.loc[mask, f"log_{y}"]
        else:
            _x = self.table.loc[mask, x]
            _y = self.table.loc[mask, y]
        stat_fun = stat_interval(_x, _y, itype=itype, conf_lev=conf_lev)
        if fit_coeffs is None:
            rmse, slope, intercept = vertical_RMSE(_x, _y, return_linregress=True)
            _logger.info(
                f"Slope is {slope:.3e} and intercept is {intercept:.3e} for linear regression fit"
            )
        else:
            slope = fit_coeffs["slope"]
            intercept = fit_coeffs["intercept"]

        # add scatter plot of data
        ax = self.scatter(x, y, ax=ax, scatter_kwargs=scatter_kwargs)
        xhat = xhat_method(
            self.table.loc[mask, x].min(), self.table.loc[mask, x].max(), 1000
        )

        if fit_in_log:
            xhat = np.log10(xhat)
        yhat = slope * xhat + intercept
        y1 = yhat - stat_fun(xhat)
        y2 = yhat + stat_fun(xhat)
        if fit_in_log:
            yhat = 10**yhat
            y1 = 10**y1
            y2 = 10**y2
            xhat = 10**xhat
        line1 = ax.plot(xhat, yhat, zorder=1)
        ax.fill_between(
            xhat,
            y1=y1,
            y2=y2,
            fc=line1[-1].get_color(),
            alpha=0.3,
            zorder=1,
            label=(
                r"$1\sigma\mathrm{-confidence}$"
                if itype == "conf"
                else r"$1\sigma\mathrm{-predictive}$"
            ),
        )
        return ax
