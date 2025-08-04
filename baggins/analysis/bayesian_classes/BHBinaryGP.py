import numpy as np
import matplotlib.pyplot as plt
import ketjugw
from arviz import plot_hdi
from baggins.analysis.bayesian_classes.GaussianProcesses import _GPBase, get_stan_file
from baggins.analysis.analyse_ketju import get_bound_binary
from baggins.env_config import _cmlogger
from baggins.plotting import savefig
from baggins.utils import get_ketjubhs_in_dir


__all__ = ["BHBinaryGP"]

_logger = _cmlogger.getChild(__name__)


class BHBinaryGP(_GPBase):
    def __init__(self, figname_base, rng=None):
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qty_labs = [r"$t/\mathrm{Myr}$"]
        self._folded_qtys_labs = [r"$a/\mathrm{pc}$"]
        self._num_OOS = 2000

    @property
    def input_qtys_labs(self):
        return self._input_qtys_labs

    @property
    def folded_qtys_labs(self):
        return self._folded_qtys_labs

    def extract_data(self, d=None):
        d = self._get_data_dir(d)
        try:
            fnames = get_ketjubhs_in_dir(d, ext=".txt")
        except NotADirectoryError:
            # the individual file names are saved to the input_data_*.yml file
            fnames = d
        except TypeError:
            fnames = d[0]
        obs = {"t": [], "a": [], "dadt_gw": [], "dedt_gw": []}
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            bh1, bh2, *_ = get_bound_binary(f)
            pars = ketjugw.orbital_parameters(bh1, bh2)
            obs["t"].append(pars["t"] / ketjugw.units.yr)
            obs["a"].append(pars["a"] / ketjugw.units.pc)
            _dadt, _dedt = ketjugw.peters_derivatives(
                pars["a"], pars["e"], pars["m0"], pars["m1"]
            )
            obs["dadt_gw"].append(_dadt / (ketjugw.units.pc / ketjugw.units.yr))
            obs["dedt_gw"].append(_dedt)
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        self.obs = obs
        self.collapse_observations(["t", "a", "dadt_gw"])

    def _set_stan_data_OOS(self):
        _OOS = {"N2": self._num_OOS}
        _t = np.linspace(
            max([min(ti) for ti in self.obs["t"]]),
            min([max(ti) for ti in self.obs["t"]]),
            _OOS["N2"],
        )
        _OOS["x2"] = _t
        self.stan_data.update(_OOS)

    def set_stan_data(self, *pars):
        super().set_stan_data(*pars)
        self.stan_data.update(
            {"x1": self.obs_collapsed["t"], "y1": self.obs_collapsed["a"]}
        )

    def all_plots(self, figsize=None):
        """
        All standard plots for a model

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["a"], 0.01),
            np.quantile(self.obs_collapsed["a"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_yscale("log")
        ax.set_ylim(*ylims)
        ax.set_xlabel(self.input_qtys_labs[0])
        ax.set_ylabel(self.folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="x1",
            ymodel=self.folded_qtys_posterior[0],
            xobs="t",
            yobs="a",
            ax=ax,
        )

        # OOS
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_yscale("log")
        ax.set_ylim(*ylims)
        ax.set_xlabel(self.input_qtys_labs[0])
        ax.set_ylabel(self.folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

    def plot_contributions(
        self, figsize=None, levels=None, smooth_kwargs=None, save=True
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        dadt_total = self.sample_generated_qty("y", state="OOS")
        dadt_gw = np.interp(
            self.stan_data["x2"], self.obs_collapsed["t"], self.obs_collapsed["dadt_gw"]
        )
        dadt_stellar = dadt_total - dadt_gw
        if levels is None:
            levels = self._default_hdi_levels
        levels.sort(reverse=True)
        cmapper, sm = self._make_default_hdi_colours(levels)
        for lev in levels:
            _logger.debug(f"Fitting level {lev}")
            plot_hdi(
                dadt_gw,
                dadt_stellar,
                hdi_prob=lev / 100,
                ax=ax,
                plot_kwargs={"c": cmapper(lev)},
                fill_kwargs={
                    "color": cmapper(lev),
                    "alpha": 0.8,
                    "label": f"{lev}% HDI",
                    "edgecolor": None,
                },
                smooth=True,
                smooth_kwargs=smooth_kwargs,
                hdi_kwargs={"skipna": True},
            )
        if save:
            savefig(
                self._make_fig_name(
                    self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"
                ),
                fig=fig,
            )
            self._gq_distribution_plot_counter += 1
        return ax

    @classmethod
    def load_fit(
        cls,
        fit_files,
        figname_base,
        rng,
    ):
        """
        Restore a stan model from a previously-saved set of csv files

        Parameters
        ----------
        model_file : str
            path to .stan file specifying the likelihood model
        fit_files : str, path-like
            path to previously saved csv files
        figname_base : str
            path-like base name that all plots will share
        rng : np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        C = super().load_fit(fit_files, figname_base, rng)
        return C
