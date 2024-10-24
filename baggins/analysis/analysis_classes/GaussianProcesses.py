from abc import abstractmethod
import numpy as np
import scipy.stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from ketjugw.units import km_per_s
from analysis.analysis_classes.StanModel import HierarchicalModel_2D
from analysis.analyse_ketju import get_bound_binary
from env_config import _cmlogger
from general.units import kpc
from literature import (
    zlochower_dry_spins,
    ketju_calculate_bh_merger_remnant_properties,
)
from mathematics import uniform_sample_sphere, convert_spherical_to_cartesian
from plotting import savefig
from utils import save_data, load_data, get_ketjubhs_in_dir


__all__ = ["_GPBase", "VkickCoreradiusGP"]

_logger = _cmlogger.getChild(__name__)


class _GPBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng) -> None:
        """
        Base class for Gaussian processes.

        Parameters
        ----------
        See input to HierarchicalModel_2D.
        Note that the class requires an RNG object to be given, as OOS
        quantities are fit in the model() section of the Stan code, making the
        model unable to be run for differing inputs when loading from a set of
        saved .csv files.
        """
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["rho", "alpha", "sigma"]
        self._latent_qtys_labs = [r"$\rho$", r"$\alpha$", r"$\sigma$"]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._folded_qtys = ["y1"]
        self._folded_qtys_posterior = ["y"]

    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def folded_qtys(self):
        return self._folded_qtys

    @property
    def folded_qtys_posterior(self):
        return self._folded_qtys_posterior

    @abstractmethod
    def extract_data(self):
        return super().extract_data()

    @abstractmethod
    def _set_stan_data_OOS(self):
        return super()._set_stan_data_OOS()

    @abstractmethod
    def set_stan_data(self):
        self.stan_data = dict(
            N1=self.num_obs_collapsed,
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()

    def sample_model(self, sample_kwargs=..., diagnose=True):
        super().sample_model(sample_kwargs, diagnose)
        if self._loaded_from_file:
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()

    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        v = super().sample_generated_quantity(gq, force_resample, state)
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state)
            return v[..., idxs]
        else:
            return v

    def plot_latent_distributions(self, figsize=None):
        """
        Plot distributions of the latent parameters of the model

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        self.plot_generated_quantity_dist(
            self._latent_qtys, ax=ax, xlabels=self._latent_qtys_labs
        )
        return ax

    def diag_plots(self, figsize=None):
        """
        Plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        type_str = "prior" if self._fit is None else "posterior"

        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent, figsize=figsize
        )

        # latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=figsize,
            labeller=self._labeller_latent,
            combine_dims={"dim"},
        )
        fig1 = ax1[0, 0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base,
                f"corner_{type_str}_{self._parameter_corner_plot_counter}",
            ),
            fig=fig1,
        )

    @abstractmethod
    def all_plots(self, figsize=None):
        self.diag_plots(figsize=figsize)
        pass


class VkickCoreradiusGP(_GPBase):
    def __init__(
        self,
        model_file,
        prior_file,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._input_qtys_labs = [r"$v/v_\mathrm{esc}$"]
        self._folded_qtys_labs = [r"$r_\mathrm{b}/r_{\mathrm{b},0}$"]
        self.escape_vel = escape_vel
        self.premerger_ketjufile = premerger_ketjufile
        self.bh1 = None
        self.bh2 = None
        self._rb0 = np.nan

    def extract_data(self, d=None, npoints=50):
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
        pars : None
            included for compatability with inherited method of StanModel()
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
        bh1.v /= km_per_s
        bh2.v /= km_per_s
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
        t, p = uniform_sample_sphere(_OOS["N2"] * 2, rng=self._rng)
        spin_mag = scipy.stats.beta.rvs(
            *zlochower_dry_spins.values(),
            random_state=self._rng,
            size=_OOS["N2"] * 2,
        )
        spins = convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
        s1 = spins[: _OOS["N2"], :]
        s2 = spins[_OOS["N2"] :, :]
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

    def save_gp_for_plots(self, fname):
        data = dict(
            vkick=self.stan_data["x1"],
            rb=self.sample_generated_quantity(
                self.folded_qtys_posterior[0], state="OOS"
            ),
        )
        save_data(data, fname)

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
