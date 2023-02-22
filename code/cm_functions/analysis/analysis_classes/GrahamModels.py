import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_2D
from . import HMQuantitiesData
from ...mathematics import get_histogram_bin_centres
from ...env_config import _cmlogger

__all__ = ["GrahamModelSimple", "GrahamModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _GrahamModelBase(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["r_b", "Re", "log10_I_b", "g", "n", "a"]
        self._latent_qtys_labs = [r"$r_\mathrm{b}/\mathrm{kpc}$", r"$R_\mathrm{e}/\mathrm{kpc}$", r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$", r"$\gamma$", r"$n$", r"$a$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
    

    @property
    def latent_qtys(self):
        return self._latent_qtys
    
    def extract_data(self, dir, pars):
        """
        Data extraction and manipulation required for the Graham density model

        Parameters
        ----------
        dir : str
            directory where hierarchical model data (from simulations) is 
        pars : dict
            analysis parameters
        """
        obs = {"R":[], "proj_density":[]}
        for f in dir:
            _logger.logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesData.load_from_file(f)
            try:
                idx = hmq.get_idx_in_vec(pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis_of_snapshot)
            except ValueError:
                _logger.logger.warning(f"No snapshot data prior to merger! The semimajor_axis_of_snapshot attribute is: {hmq.semimajor_axis_of_snapshot}. This run will not form part of the analysis.")
                continue
            except AssertionError:
                _logger.logger.warning(f"Trying to search for value {pars['bh_binary']['target_semimajor_axis']['value']}, but an AssertionError was thrown. The array bounds are {min(hmq.semimajor_axis_of_snapshot)} - {max(hmq.semimajor_axis_of_snapshot)}. This run will not form part of the analysis.")
                continue
            r = get_histogram_bin_centres(hmq.radial_edges)
            obs["R"].append(r)
            obs["proj_density"].append(list(hmq.projected_mass_density.values())[idx])
        self.obs = obs

        self.transform_obs("R", "log10_R", lambda x: np.log10(x))
        self.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))
        self.transform_obs("log10_proj_density", "log10_proj_density_mean", lambda x: np.nanmean(x, axis=0))
        self.transform_obs("log10_proj_density", "log10_proj_density_std", lambda x: np.nanstd(x, axis=0))
        self.collapse_observations(["R", "log10_R", "log10_proj_density_mean", "log10_proj_density_std"])
    

    def plot_latent_distributions(self, figsize=None):
        """
        Plot distributions of the latent parameters of the model

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            plotting axis
        """
        fig, ax = plt.subplots(2,3, figsize=figsize)
        ax = np.concatenate(ax).flatten()
        ax[3].set_xscale("log")
        self.plot_generated_quantity_dist(self._latent_qtys, ax=ax, xlabels=self._latent_qtys_labs)
        return ax


    def all_prior_plots(self, figsize=None):
        """
        Prior plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        # prior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_ylim(-1, 15.1)
        ax1.set_xlabel("R/kpc")
        ax1.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
        ax1.set_xscale("log")
        self.prior_plot("R", "log10_proj_density_mean", xmodel="R", ymodel="log10_surf_rho_prior", yobs_err="log10_proj_density_std", ax=ax1)

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)



class GrahamModelSimple(_GrahamModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.figname_base = f"{self.figname_base}-simple"

    def all_posterior_plots(self, figsize=None):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            plotting axis
        """
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self.latent_qtys, labeller=self._labeller_latent)

        # posterior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"log($R$/kpc)")
        ax1.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
        ax1.set_ylim(6, 10)
        self.posterior_plot("log10_R", "log10_proj_density_mean", "log10_surf_rho_posterior", yobs_err="log10_proj_density_std", ax=ax1)

        # latent parameter distributions
        ax = self.plot_latent_distributions(figsize=figsize)
        return ax



class GrahamModelHierarchy(_GrahamModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
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
        ax : matplotlib.axes._subplots.AxesSubplot
            plotting axis
        """
        # latent parameter plots (corners, chains, etc)
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


