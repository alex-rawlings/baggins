import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_2D, HMQuantitiesData
from ...env_config import _cmlogger
from ...plotting import savefig


__all__ = ["QuinlanModelSimple", "QuinlanModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _QuinlanModelBase(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["HGp_s", "inv_a_0", "K", "e0"]
        #self._latent_qtys_labs = [r"$HG\rho/\sigma / (\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$", r"$\mathrm{kpc}/a_0$", r"$K$", r"$e_0$"]
        self._latent_qtys_labs = [r"$H'(\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$", r"$\mathrm{kpc}/a_0$", r"$K$", r"$e_0$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))

    @property
    def latent_qtys(self):
        return self._latent_qtys


    def extract_data(self, dir, pars):
        obs = {"t":[], "a":[], "e":[], "e_ini":[]}
        i = 0
        for f in dir:
            _logger.logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesData.load_from_file(f)
            status, idx0 = hmq.idx_finder(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
            if not status: continue
            status, idx1 = hmq.idx_finder(pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis)
            if not status: continue
            try:
                assert idx0 < idx1
            except AssertionError:
                _logger.logger.exception(f"Lower index {idx0} is not less than upper index {idx1}!", exc_info=True)
                raise
            idxs = np.r_[idx0:idx1]
            obs["t"].append(hmq.binary_time[idxs])
            obs["a"].append(hmq.semimajor_axis[idxs])
            obs["e"].append(hmq.eccentricity[idxs])
            try:
                obs["e_ini"].append([hmq.initial_galaxy_orbit["e0"]])
            except AttributeError:
                _logger.logger.warning(f"File {f} is missing the 'initial_galaxy_orbit' attribute. Ideally, re-run HMQ extraction process.")

            i += 1
        if not obs["e_ini"]:
            obs.pop("e_ini")
        self.obs = obs

        # some transformations we need
        self.transform_obs("a", "inva", lambda x:1/x)
        self.transform_obs("t", "t_shift", lambda x:x-x[0])
        self.collapse_observations(["t_shift", "inva", "e"])


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
        fig, ax = plt.subplots(2, 2, figsize=figsize)
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
        # rename dimensions for collapsing
        self.rename_dimensions({"HGp_s_dim_0":"group", "inv_a_0_dim_0":"group", "K_dim_0":"group", "e0_dim_0":"group"})

        fig, ax = plt.subplots(2,1, figsize=figsize,sharex="all")
        ax[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        ax[0].set_ylabel(r"$\mathrm{kpc}/a$")
        ax[1].set_ylabel(r"$e$")
        self.prior_plot(xobs="t_shift", yobs="inva", xmodel="t", ymodel="inv_a_prior", ax=ax[0], save=False)
        self.prior_plot(xobs="t_shift", yobs="e", xmodel="t", ymodel="e_prior", ax=ax[1], show_legend=False)
        
        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        fig1 = ax1[0,0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"), fig=fig1)


    def determine_merger_timescale_distribution(self, n=1000):
        cumulative_marginals_list = []
        num_latents = len(self.latent_qtys)
        for i, l in enumerate(self.latent_qtys):
            y = self.sample_generated_quantity(l)
            cumulative_marginals_list.append(
                np.cumsum(y)
            )
        timescales = np.full(n, np.nan)
        for i in range(n):
            # draw initial state from distribution
            ic = np.full(num_latents, np.nan)
            for j in range(num_latents):
                ic[j] = cumulative_marginals_list[j][self._rng.uniform(0,1)]
            # TODO dict instead of list so ordering isn't important
            # plug IC into peter evolution
            # don't forget inva transform







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
        fig1, ax1 = plt.subplots(2,1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        ax1[0].set_ylabel(r"$mathrm{kpc}/a$")
        ax1[1].set_ylabel(r"$e$")
        self.posterior_plot(xobs="t", yobs="inva", ymodel="inv_a_posterior", ax=ax1[0], save=False)
        self.posterior_plot(xobs="t", yobs="e", ymodel="ecc_posterior", ax=ax1[1], show_legend=False)

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
        self._hyper_qtys = ["HGp_s_mean", "HGp_s_std", "inv_a_0_mean", "inv_a_0_std", "K_mean", "K_std", "e0_mean", "e0_std"]
        #self._hyper_qtys_labs = [r"$\mu_{HG\rho/\sigma} / (\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$", r"$\sigma_{HG\rho/\sigma} / (\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$", r"$\mu_{1/a_0} / \mathrm{kpc}^{-1}$", r"$\sigma_{1/a_0} / \mathrm{kpc}^{-1}$", r"$\mu_K$", r"$\sigma_K$", r"$\mu_{e_0}$", r"$\sigma_{e_0}$"]
        self._hyper_qtys_labs = [r"$\mu_{H'}$", r"$\sigma_{H'}$", r"$\mu_{1/a_0}$", r"$\sigma_{1/a_0}$", r"$\mu_K$", r"$\sigma_K$", r"$\mu_{e_0}$", r"$\sigma_{e_0}$"]
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
        self.rename_dimensions({"HGp_s_dim_0":"group", "inv_a_0_dim_0":"group", "K_dim_0":"group", "e0_dim_0":"group"})

        # hyperparameter plots
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive checks
        fig1, ax1 = plt.subplots(2,1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        ax1[0].set_ylabel(r"$\mathrm{kpc}/a$")
        ax1[1].set_ylabel(r"$e$")
        self.posterior_plot(xobs="t_shift", yobs="inva", xmodel="t", ymodel="inv_a_posterior", ax=ax1[0], save=False)
        self.posterior_plot(xobs="t_shift", yobs="e", xmodel="t", ymodel="ecc_posterior", ax=ax1[1], show_legend=False)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax
