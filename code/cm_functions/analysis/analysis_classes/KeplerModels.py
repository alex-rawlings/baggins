import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_1D, HMQuantitiesData
from ..orbit import find_idxs_of_n_periods
from ...env_config import _cmlogger
from ...plotting import savefig
from ketjugw.units import unit_length_in_pc, unit_time_in_years

__all__ = ["KeplerModelSimple", "KeplerModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _KeplerModelBase(StanModel_1D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["a_hard", "e_hard"]
        self._latent_qtys_labs = [r"$a_\mathrm{h}/\mathrm{pc}$", "$e_\mathrm{h}$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
    
    @property
    def latent_qtys(self):
        return self._latent_qtys
    
    def extract_data(self, dir, pars):
        """
        Extract data and manipulations required for the Kepler binary model

        Parameters
        ----------
        dir : str
            directory where hierarchical model data (from simulations) is
        pars : dict
            analysis parameters
        """
        obs = {"angmom":[], "a":[], "e":[], "mass1":[], "mass2":[]}
        i = 0
        for f in dir:
            _logger.logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesData.load_from_file(f)
            try:
                idx = hmq.get_idx_in_vec(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
            except ValueError:
                _logger.logger.warning(f"No data prior to merger! The requested semimajor axis value is {np.nanmedian(hmq.hardening_radius)}, semimajor_axis attribute is: {hmq.semimajor_axis}. This run will not form part of the analysis.")
                continue
            except AssertionError:
                _logger.logger.warning(f"Trying to search for value {np.nanmedian(hmq.hardening_radius)}, but an AssertionError was thrown. The array bounds are {min(hmq.semimajor_axis)} - {max(hmq.semimajor_axis)}. This run will not form part of the analysis.")
                continue
            t_target = hmq.binary_time[idx]
            _logger.logger.debug(f"Target time: {t_target} Myr")
            try:
                target_idx, delta_idxs = find_idxs_of_n_periods(t_target, hmq.binary_time, hmq.binary_separation, num_periods=pars["bh_binary"]["num_orbital_periods"])
            except IndexError:
                _logger.logger.warning(f"Orbital period for hard semimajor axis not found! This run will not form part of the analysis.")
                continue
            _logger.logger.debug(f"For observation {i} found target time between indices {delta_idxs[0]} and {delta_idxs[1]}")
            period_idxs = np.r_[delta_idxs[0]:delta_idxs[1]]
            obs["angmom"].append(hmq.binary_angular_momentum[period_idxs])
            # convert semimajor axis from kpc to pc here
            obs["a"].append(hmq.semimajor_axis[period_idxs] * 1000)
            obs["e"].append(hmq.eccentricity[period_idxs])
            try:
                obs["mass1"].append([hmq.binary_masses[0]])
                obs["mass2"].append([hmq.binary_masses[1]])
            except AttributeError:
                _logger.logger.error(f"Attribute 'particle_masses' does not exist for {f}! Will assume equal-mass BHs!")
                obs["mass1"].append([hmq.masses_in_galaxy_radius["bh"][0]/2])
                obs["mass2"].append([hmq.masses_in_galaxy_radius["bh"][0]/2])
            i += 1
        self.obs = obs

        # some transformations we need
        G_in_Msun_pc_yr = unit_length_in_pc**3 / unit_time_in_years**2
        self.transform_obs(("mass1", "mass2"), "total_mass", lambda x,y: x+y)
        self.obs["total_mass_long"] = []
        for tm, am in zip(self.obs["total_mass"], self.obs["angmom"]):
            self.obs["total_mass_long"].append(np.repeat(tm, len(am)))
        self.transform_obs("angmom", "angmom_corr", lambda x: x*unit_length_in_pc**2/unit_time_in_years)
        self.transform_obs(("angmom_corr", "total_mass_long"), "angmom_corr_red", lambda l, m: l/np.sqrt(G_in_Msun_pc_yr * m))
        self.transform_obs("angmom_corr_red", "log10_angmom_corr_red", lambda x: np.log10(x))
        self.transform_obs("total_mass_long", "log10_total_mass_long", lambda x: np.log10(x))
        self.collapse_observations(["log10_angmom_corr_red", "a", "e", "total_mass_long", "log10_total_mass_long"])


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
        ax.set_xlabel(r"$\log\left( \left(l/\sqrt{GM}\right)/\sqrt{\mathrm{pc}} \right)$")
        ax.set_ylabel("PDF")
        self.prior_plot(xobs="log10_angmom_corr_red", xmodel="log10_angmom", ax=ax)
        
        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)




class KeplerModelSimple(_KeplerModelBase):
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
        ax1.set_xlabel(r"$\log\left( \left(l/\sqrt{GM}\right)/\sqrt{\mathrm{pc}} \right)$")
        ax1.set_ylabel("PDF")
        self.posterior_plot(xobs="log10_angmom_corr_red", xmodel="log10_angmom_posterior", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)
        
        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax




class KeplerModelHierarchy(_KeplerModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.figname_base = f"{self.figname_base}-hierarchy"
        self._hyper_qtys = ["a_hard_mu", "a_hard_sigma", "e_hard_mu", "e_hard_sigma"]
        self._hyper_qtys_labs = [r"$\mu_{a_\mathrm{h}}$", r"$\sigma_{a_\mathrm{h}}$", r"$\mu_{e_\mathrm{h}}$", r"$\sigma_{e_\mathrm{h}}$"]
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
        self.rename_dimensions({"a_hard_dim_0":"group", "e_hard_dim_0":"group"})

        # hyper parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive checks
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"$\log\left( \left(l/\sqrt{GM}\right)/\sqrt{\mathrm{pc}} \right)$")
        ax1.set_ylabel("PDF")
        self.posterior_plot(xobs="log10_angmom_corr_red", xmodel="log10_angmom_posterior", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)
        # append a boundary value of e to data? Ensures 
        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        #ax[1,0].set_ylim(0,1)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax

