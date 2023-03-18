import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_2D, HMQuantitiesData
from ...env_config import _cmlogger
from ...general import get_idx_in_array
from ...plotting import savefig


__all__ = ["QuinlanModelSimple", "QuinlanModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _QuinlanModelBase(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["HGp_s", "inv_a_0"]
        self._latent_qtys_labs = [r"$HG\rho/\sigma$", r"$a_0^{-1}$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))

    @property
    def latent_qtys(self):
        return self._latent_qtys
    
    def _idx_finder(self, val, vec):
        try:
            idx = get_idx_in_array(val, vec)
            status = True
        except ValueError:
            _logger.logger.warning(f"No data prior to merger! The requested semimajor axis value is {val}, semimajor_axis attribute is: {vec}. This run will not form part of the analysis.")
            status = False
            idx = -9999
        except AssertionError:
            _logger.logger.warning(f"Trying to search for value {val}, but an AssertionError was thrown. The array bounds are {min(vec)} - {max(vec)}. This run will not form part of the analysis.")
            status = False
            idx = -9999
        return status, idx

    
    def extract_data(self, dir, pars):
        obs = {"t":[], "a":[]}
        i = 0
        for f in dir:
            _logger.logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesData.load_from_file(f)
            status, idx0 = self._idx_finder(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
            if not status: continue
            status, idx1 = self._idx_finder(pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis)
            if not status: continue
            try:
                assert idx0 < idx1
            except AssertionError:
                _logger.logger.exception(f"Lower index {idx0} is not less than upper index {idx1}!", exc_info=True)
                raise
            idxs = np.r_[idx0:idx1]
            obs["t"].append(hmq.binary_time[idxs])
            obs["a"].append(hmq.semimajor_axis[idxs])
            i += 1
        self.obs = obs

        # some transformations we need
        self.transform_obs("a", "inva", lambda x:1/x)
        self.collapse_observations(["t", "inva"])


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
        fig, ax = plt.subplots(1, 2, figsize=figsize)
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
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_xlabel(r"$t/\mathrm{Myr}$")
        ax.set_ylabel(r"$\mathrm{pc}/a$")
        self.prior_plot(xobs="t", yobs="inva", xmodel="t", ymodel="inv_a_prior", ax=ax)
        
        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)




class QuinlanModelSimple(_QuinlanModelBase):
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
        ax : np.ndarray
            array of plotting axes for latent parameter corner plot
        """
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self.latent_qtys, labeller=self._labeller_latent)

        # posterior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"$t/\mathrm{Myr}$")
        ax1.set_ylabel(r"$mathrm{pc}/a$")
        self.posterior_plot(xobs="t", yobs="inva", ymodel="inv_a_posterior", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)
        
        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax




class QuinlanModelHierarchy(_QuinlanModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.figname_base = f"{self.figname_base}-hierarchy"
        self._hyper_qtys = ["HGp_s_mean", "HGp_s_std", "inv_a_0_mean", "inv_a_0_std"]
        self._hyper_qtys_labs = [r"$\mu_{HG\rho/\sigma}$", r"$\sigma_{HG\rho/\sigma}$", r"$\mu_{1/a_0}$", r"$\sigma_{1/a_0}$"]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qtys_labs)))


    def all_posterior_plots(self, figsize=None):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : np.ndarray
            array of plotting axes for latent parameter corner plot
        """
        # rename dimensions for collapsing
        self.rename_dimensions({"HGp_s_dim_0":"group", "inv_a_0_dim_0":"group"})

        # hyperparameter plots
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive checks
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"$t/\mathrm{Myr}$")
        ax1.set_ylabel(r"$\mathrm{pc}/a$")
        self.posterior_plot(xobs="t", yobs="inva", xmodel="t", ymodel="inv_a_posterior", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax
