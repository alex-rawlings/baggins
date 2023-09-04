from abc import abstractmethod
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_2D
from . import HMQuantitiesData
from ...mathematics import get_histogram_bin_centres
from ...env_config import _cmlogger
from ...plotting import savefig
from ...utils import get_files_in_dir

__all__ = ["GrahamModelSimple", "GrahamModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _GrahamModelBase(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self._folded_qtys = ["log10_surf_rho"]
        self._folded_qtys_labs = [r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["r_b", "Re", "log10_I_b", "g", "n", "a", "err"]
        self._latent_qtys_posterior = [f"{v}_posterior" if "err" not in v else v for v in self.latent_qtys]
        self._latent_qtys_labs = [r"$r_\mathrm{b}/\mathrm{kpc}$", r"$R_\mathrm{e}/\mathrm{kpc}$", r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$", r"$\gamma$", r"$n$", r"$a$", r"$\sigma$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
        self._labeller_latent_posterior = MapLabeller(dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs)))
        self._merger_id = None


    @property
    def folded_qtys(self):
        return self._folded_qtys

    @property
    def folded_qtys_posterior(self):
        return self._folded_qtys_posterior

    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    @property
    def merger_id(self):
        return self._merger_id


    @abstractmethod
    def extract_data(self, pars, d=None):
        """
        Data extraction and manipulation required for the Graham density model

        Parameters
        ----------
        pars : dict
            analysis parameters
        d : path-like, optional
            HMQ data directory, by default None (paths read from 
            `_input_data_files`)
        """
        obs = {"R":[], "proj_density":[]}
        d = self._get_data_dir(d)
        if self._loaded_from_file:
            fnames = d[0]
        else:
            fnames = get_files_in_dir(d)
            _logger.logger.debug(f"Reading from dir: {d}")
        for f in fnames:
            _logger.logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesData.load_from_file(f)
            status, idx = hmq.idx_finder(pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis)
            if not status: continue
            r = get_histogram_bin_centres(hmq.radial_edges)
            obs["R"].append(r)
            obs["proj_density"].append(list(hmq.projected_mass_density.values())[idx])
            if self._merger_id is None:
                self._merger_id = re.sub("_[a-z]-", "-", hmq.merger_id)
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        self.obs = obs

        # some transformations we need
        self.transform_obs("R", "log10_R", lambda x: np.log10(x))
        self.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))
        self.transform_obs("log10_proj_density", "log10_proj_density_mean", lambda x: np.nanmean(x, axis=0))
        self.transform_obs("log10_proj_density", "log10_proj_density_std", lambda x: np.nanstd(x, axis=0))
        self.collapse_observations(["R", "log10_R", "log10_proj_density_mean", "log10_proj_density_std"])


    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables
        """
        try:
            assert self.num_OOS is not None
        except AssertionError:
            _logger.logger.exception(f"num_OOS cannot be None when setting Stan data!", exc_info=True)
            raise
        self.stan_data["N_OOS"] = self.num_OOS
        self.stan_data["group_id_OOS"] = self._rng.integers(1, self.num_groups, size=self.num_OOS, endpoint=True)
        self.stan_data["R_OOS"] = np.linspace(
                            np.max([R[0] for R in self.obs["R"]]),
                            np.min([R[-1] for R in self.obs["R"]]),
                            self.num_OOS
        )


    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling
        """
        self.stan_data = dict(
            N_tot = self.num_obs,
            N_groups = self.num_groups,
            group_id = self.obs_collapsed["label"],
            R = self.obs_collapsed["R"],
            log10_surf_rho = self.obs_collapsed["log10_proj_density_mean"],
            log10_surf_rho_err = self.obs_collapsed["log10_proj_density_std"]
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()


    def sample_model(self, sample_kwargs={}):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS 
        from previous sample.
        """
        super().sample_model(sample_kwargs)
        if self._loaded_from_file:
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()


    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        v = super().sample_generated_quantity(gq, force_resample, state)
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state)
            return v[...,idxs]
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
        fig, ax = plt.subplots(2,3, figsize=figsize)
        try:
            self.plot_generated_quantity_dist(self.latent_qtys_posterior, ax=ax, xlabels=self._latent_qtys_labs)
        except ValueError:
            self.plot_generated_quantity_dist(self.latent_qtys, ax=ax, xlabels=self._latent_qtys_labs)
        ax[1,0].set_xscale("log")
        return ax


    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        """
        Prior plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            figure y-limits, by default (-1, 15.1)
        """
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys if "err" not in k]), "group")
        self._expand_dimension(["err"], "group")

        # prior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_ylim(*ylim)
        ax1.set_xlabel("R/kpc")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        self.plot_predictive(xmodel="R", ymodel=f"{self._folded_qtys[0]}_prior", xobs="R", yobs="log10_proj_density_mean", yobs_err="log10_proj_density_std", ax=ax1)

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        fig1 = ax1[0,0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"), fig=fig1)



class GrahamModelSimple(_GrahamModelBase):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self.figname_base = f"{self.figname_base}-simple"


    def extract_data(self, pars, d=None):
        """
        See docs for `_GrahamModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/quinlan-hardening-{self.merger_id}-simple")


    def all_posterior_plots(self, figsize=None, ylim=(6, 10)):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            figure y-limits, by default (6, 10)

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self.latent_qtys, labeller=self._labeller_latent)

        # posterior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"log($R$/kpc)")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_ylim(*ylim)
        # TODO scale of x axis??
        self.plot_predictive(xmodel="R", ymodel=f"{self._folded_qtys_posterior[0]}", xobs="R", yobs="log10_proj_density_mean", yobs_err="log10_proj_density_std", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax


# TODO update this class!!
class GrahamModelHierarchy(_GrahamModelBase):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self.figname_base = f"{self.figname_base}-hierarchy"
        self._hyper_qtys = ["r_b_mean", "r_b_std", "Re_mean", "Re_std", "log10_I_b_mean", "log10_I_b_std", "g_mean", "g_std", "n_mean", "n_std", "a_mean", "a_std"]
        self._hyper_qts_labs = [r"$\mu_{r_\mathrm{b}}$", r"$\sigma_{r_\mathrm{b}}$", r"$\mu_{R_\mathrm{e}}$", r"$\sigma_{R_\mathrm{e}}$", r"$\mu_{\log_{10}\Sigma_\mathrm{b}}$", r"$\sigma_{\log_{10}\Sigma_\mathrm{b}}$", r"$\mu_{g}$", r"$\sigma_{g}$", r"$\mu_{n}$", r"$\sigma_{n}$", r"$\mu_{a}$", r"$\sigma_{a}$"]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qts_labs)))


    def all_posterior_plots(self, figsize=None):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        # hyper parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(["r_b_mean", "r_b_std", "Re_mean", "Re_std"], labeller=self._labeller_hyper)
        self.parameter_diagnostic_plots(["log10_I_b_mean", "log10_I_b_std", "a_mean", "a_std"], labeller=self._labeller_hyper)
        self.parameter_diagnostic_plots(["g_mean", "g_std", "n_mean", "n_std"], labeller=self._labeller_hyper)

        # posterior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"log($R$/kpc)")
        ax1.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
        ax1.set_ylim(6, 10)
        self.posterior_plot("log10_R", "log10_proj_density_mean", "log10_surf_rho_posterior", yobs_err="log10_proj_density_std", ax=ax1)

        # latent parameter distributions
        ax = self.plot_latent_distributions(figsize=figsize)
        return ax


