import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ketjugw
from baggins.analysis.analyse_ketju import get_bound_binary
from baggins.analysis.bayesian_classes.GaussianProcesses import _GPBase, get_stan_file
from baggins.env_config import _cmlogger
from baggins.general.units import kpc
from baggins.literature import SMBHSpins, ketju_calculate_bh_merger_remnant_properties
from baggins.utils.data_handling import get_ketjubhs_in_dir, load_data


__all__ = ["VkickCoreradiusGP", "CoreradiusVkickGP"]

_logger = _cmlogger.getChild(__name__)


class VkickCoreradiusGP(_GPBase):
    def __init__(
        self,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        """
        Gaussian process regression of the kick velocity - core radius relation.

        Parameters
        ----------
        figname_base : str
            path-like base name that all plots will share
        escape_vel : float, optional
            escape velocity of the system, by default None
        premerger_ketjufile : str, optional
            ketju_bhs file that has data on the merger of the two BHs, by default None
        rng :  np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qtys_labs = [r"$v/v_\mathrm{esc}$"]
        self._folded_qtys_labs = [r"$r_\mathrm{b}/r_{\mathrm{b},0}$"]
        self.escape_vel = escape_vel
        self.premerger_ketjufile = premerger_ketjufile
        self.bh1 = None
        self.bh2 = None
        self._rb0 = np.nan

    def extract_data(self, d=None):
        """
        Data extraction and manipulation required by the CoreKick model.
        Due to the complexity of extracting core radius, samples from the core
        radius distribution for each kick velocity are assumed as an input
        pickle file. The structure of the file must be of the form:
        {'rb': {
        "XXXX": [core radius values],
        ...,
        "YYYY": [core radius values]
        }}
        Where XXXX and YYYY are the kick velocities as strings, convertible to
        a float (e.g. "0060").
        The `pars` parameter is unused and included only for compatability with
        the parent class.


        Parameters
        ----------
        d : path-like, optional
            file of core radius samples, by default None (paths read from
            `_input_data_files`)
        """
        try:
            assert self.escape_vel is not None and self.premerger_ketjufile is not None
        except AssertionError:
            _logger.exception(
                "Attributes `escape_vel` and `premerger_ketjufile` must be set before extracting data",
                exc_info=True,
            )
            raise
        d = self._get_data_dir(d)
        data = load_data(d)
        obs = {"vkick": [], "rb": []}
        for k, v in data["rb"].items():
            if k == "__githash" or k == "__script":
                continue
            _logger.info(f"Getting data for kick {k}")
            mask = ~np.isnan(v)
            v = v[mask]
            rb0 = data["rb"]["0000"][mask]
            obs["vkick"].append([float(k) / self.escape_vel])
            obs["rb"].append([np.nanmedian(v.flatten() / rb0.flatten())])
        self._rb0 = np.nanmean(rb0)
        self.obs = obs
        if not self._loaded_from_file:
            self._add_input_data_file(d)
        self.collapse_observations(["vkick", "rb"])
        # extract BH data at the timestep before merger
        kfile = get_ketjubhs_in_dir(self.premerger_ketjufile)[0]
        bh1, bh2, *_ = get_bound_binary(kfile)
        # move to Gadget units: kpc, km/s, 1e10Msol
        bh1.x /= kpc
        bh2.x /= kpc
        bh1.v /= ketjugw.units.km_per_s
        bh2.v /= ketjugw.units.km_per_s
        bh1.m /= 1e10
        bh2.m /= 1e10
        self.bh1 = bh1[-1]
        self.bh2 = bh2[-1]

    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables. 10000 OOS points will be
        used.
        BH spins are uniformly sampled on the sphere, with magnitude from the
        Zlochower Lousto "dry" distribution.
        """
        _OOS = {"N2": 1000}
        self._num_OOS = _OOS["N2"]
        spins = SMBHSpins("zlochower_dry", "uniform")
        s1 = spins.sample_spins(n=_OOS["N2"])
        s2 = spins.sample_spins(n=_OOS["N2"])
        vkick = np.full(_OOS["N2"], np.nan)
        for i, (ss1, ss2) in tqdm(enumerate(zip(s1, s2)), total=len(s1)):
            remnant = ketju_calculate_bh_merger_remnant_properties(
                m1=self.bh1.m,
                m2=self.bh2.m,
                s1=ss1,
                s2=ss2,
                x1=self.bh1.x.flatten(),
                x2=self.bh2.x.flatten(),
                v1=self.bh1.v.flatten(),
                v2=self.bh2.v.flatten(),
            )
            vkick[i] = np.linalg.norm(remnant["v"]) / self.escape_vel
        _logger.debug(
            f"{np.sum(np.isnan(vkick)) / len(vkick) * 100:.2f}% of calculations from from the Zlochower Lousto relation are NaN!"
        )
        _OOS["x2"] = vkick[~np.isnan(vkick)]
        self.stan_data.update(_OOS)

    def set_stan_data(self):
        super().set_stan_data()
        self.stan_data.update(
            dict(x1=self.obs_collapsed["vkick"], y1=self.obs_collapsed["rb"])
        )

    def all_plots(self, figsize=None):
        """
        All posterior predictive check and out-of-sample plots.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["rb"], 0.01),
            np.quantile(self.obs_collapsed["rb"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="x1",
            ymodel=self.folded_qtys_posterior[0],
            xobs="vkick",
            yobs="rb",
            ax=ax,
        )

        # OOS
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

        rb_mode = self.calculate_mode("y")
        _logger.info(
            f"Forward-folded core radius mode is {rb_mode*self._rb0:.3f} kpc ({rb_mode:.3f} rb0)"
        )

        # marginal distribution of dependent variable
        fig, ax_rb = plt.subplots()
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        ax_rb.tick_params(axis="x", which="both", top=False)
        rb02kpc = lambda x: x * self._rb0
        kpc2rb0 = lambda x: x / self._rb0
        secax = ax_rb.secondary_xaxis("top", functions=(rb02kpc, kpc2rb0))
        secax.set_xlabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
        self.plot_generated_quantity_dist(
            ["y"],
            state="OOS",
            bounds=[(0, 3)],
            xlabels=self._folded_qtys_labs,
            ax=ax_rb,
        )

    @classmethod
    def load_fit(
        cls,
        model_file,
        fit_files,
        figname_base,
        escape_vel,
        premerger_ketjufile,
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
            _description_
        escape_vel : float
            system escape velocity
        premerger_ketjufile : path-like
            ketju_bhs.hdf5 file of the pre-merger BHs
        rng : np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        C = super().load_fit(model_file, fit_files, figname_base, rng)
        C.escape_vel = escape_vel
        C.premerger_ketjufile = premerger_ketjufile
        return C


class CoreradiusVkickGP(_GPBase):
    def __init__(self, figname_base, rng) -> None:
        """
        Gaussian process regression of the core radius - kick velocity relation.

        Parameters
        ----------
        figname_base : str
            path-like base name that all plots will share
        rng :  np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qtys_labs = [r"$r_\mathrm{b}/r_{\mathrm{b},0}$"]
        self._folded_qtys_labs = [r"$v/v_\mathrm{esc}$"]
        self._rb0 = np.nan

    def extract_data(self, d=None):
        """
        Data extraction and manipulation required by the CoreKick model.
        Due to the complexity of extracting core radius, samples from the core
        radius distribution for each kick velocity are assumed as an input
        pickle file. The structure of the file must be of the form:
        {'rb': {
        "XXXX": [core radius values],
        ...,
        "YYYY": [core radius values]
        }}
        Where XXXX and YYYY are the kick velocities as strings, convertible to
        a float (e.g. "0060").


        Parameters
        ----------
        d : path-like, optional
            file of core radius samples, by default None (paths read from
            `_input_data_files`)
        """
        try:
            assert self.escape_vel is not None and self.premerger_ketjufile is not None
        except AssertionError:
            _logger.exception(
                "Attributes `escape_vel` and `premerger_ketjufile` must be set before extracting data",
                exc_info=True,
            )
            raise
        d = self._get_data_dir(d)
        data = load_data(d)
        obs = {"vkick": [], "rb": []}
        for k, v in data["rb"].items():
            if k == "__githash" or k == "__script":
                continue
            _logger.info(f"Getting data for kick {k}")
            mask = ~np.isnan(v)
            v = v[mask]
            rb0 = data["rb"]["0000"][mask]
            obs["vkick"].append([float(k)])
            # take the median core radius value for each recoil velocity
            obs["rb"].append([np.nanmedian(v.flatten() / rb0.flatten())])
        self._rb0 = np.nanmedian(rb0)
        self.obs = obs
        if not self._loaded_from_file:
            self._add_input_data_file(d)
        self.collapse_observations(["vkick", "rb"])

    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variable for core radius normalised to
        rb0 (binary-scoured radius)
        """
        _OOS = {"N2": 1000}
        self._num_OOS = _OOS["N2"]
        _OOS["x2"] = np.linspace(1, np.max(self.obs_collapsed["rb"]), self.num_OOS)
        self.stan_data.update(_OOS)

    def set_stan_data(self):
        super().set_stan_data()
        self.stan_data.update(
            dict(x1=self.obs_collapsed["rb"], y1=self.obs_collapsed["vkick"])
        )

    def all_plots(self, figsize=None):
        """
        All posterior predictive check and out-of-sample plots.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["vkick"], 0.01),
            np.quantile(self.obs_collapsed["vkick"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="x1",
            ymodel=self.folded_qtys_posterior[0],
            xobs="rb",
            yobs="vkick",
            ax=ax,
        )

        # OOS
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

        vkick_mode = self.calculate_mode("y")
        _logger.info(f"Forward-folded kick velocity mode is {vkick_mode:.3f} km/s")

        # marginal distribution of dependent variable
        fig, ax_vk = plt.subplots()
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        self.plot_generated_quantity_dist(
            ["y"],
            state="OOS",
            bounds=[(0, 400)],
            xlabels=self._folded_qtys_labs,
            ax=ax_vk,
        )
