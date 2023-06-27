import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_2D, HMQuantitiesData
from .. import first_major_deflection_angle, find_idxs_of_n_periods
from ...env_config import _cmlogger
from ...plotting import savefig
from ...utils import get_files_in_dir


__all__ = ["DeflectionAngleGP"]

_logger = _cmlogger.copy(__file__)



class DeflectionAngleGP(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["rho", "alpha", "sigma", "eta"]
        self._latent_qtys_labs = [r"$\rho$", r"$\alpha$", r"$\sigma$", r"$\eta$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
        self._e_ini = None


    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def e_ini(self):
        return self._e_ini


    def extract_data(self, pars, dirs=None, tol=1e-5):
        """
        Extract data from HMQcubes required for analysis

        Parameters
        ----------
        pars : dict
            analysis parameters
        dirs : list
            HMQ data directories to include, by default None (paths read from 
            `_input_data_files`)
        tol : float
            relative tolerance for merger orbit consistency, by default 1e-5
        """
        if dirs is None:
            try:
                assert self._loaded_from_file
                dirs = [[f["path"] for f in self._input_data_files.values()]]
            except AssertionError:
                _logger.logger.exception(f"Parameter 'dirs' must be given if not loaded from file!", exc_info=True)
                raise
        obs = {"theta":[], "a":[], "e":[]}
        # dummy variables for consistency checks between directories
        ini_merger = {}
        for d in dirs:
            if self._loaded_from_file:
                fnames = d
            else:
                fnames = get_files_in_dir(d)
            for f in fnames:
                _logger.logger.info(f"Loading file: {f}")
                hmq = HMQuantitiesData.load_from_file(f)
                # protect against instances where no data for bound binary
                try:
                    hmq.hardening_radius
                except AttributeError:
                    _logger.logger.warning(f"No attribute `hardening_radius` for file {f}: skipping.")
                    continue
                theta = first_major_deflection_angle(hmq.prebound_deflection_angles)[0]
                if np.isnan(theta):
                    _logger.logger.warning(f"No prebound deflection angles in file {f}: skipping.")
                    continue
                a = np.nanmedian(hmq.hardening_radius)
                status, hard_idx = hmq.idx_finder(a, hmq.semimajor_axis)
                if not status:
                    _logger.logger.warning(f"No semimajor axis values correspond to hardening radius in file {f}: skipping.")
                    continue
                t_target = hmq.binary_time[hard_idx]
                _logger.logger.debug(f"Target time: {t_target} Myr")
                try:
                    target_idx, delta_idxs = find_idxs_of_n_periods(t_target, hmq.binary_time, hmq.binary_separation, num_periods=pars["bh_binary"]["num_orbital_periods"])
                except IndexError:
                    _logger.logger.warning(f"Orbital period for hard semimajor axis not found! This run will not form part of the analysis.")
                    continue
                _logger.logger.debug(f"For observation {self._input_data_file_count} found target time between indices {delta_idxs[0]} and {delta_idxs[1]}")
                period_idxs = np.r_[delta_idxs[0]:delta_idxs[1]]
                obs["theta"].append([theta])
                obs["a"].append([a])
                obs["e"].append([np.nanmedian(hmq.eccentricity[period_idxs])])

                # ensure same merger orbit
                if self._input_data_file_count == 0:
                    for k, v in hmq.initial_galaxy_orbit.items():
                        ini_merger[k] = v
                else:
                    for k, v in ini_merger.items():
                        try:
                            assert np.abs(hmq.initial_galaxy_orbit[k] - v)/v < tol
                        except AssertionError:
                            _logger.logger.exception(f"File {f} has an inconsistent merger orbit! Value of {k} is {hmq.initial_galaxy_orbit[k]}, should be {v}!", exc_info=True)
                            raise
                if not self._loaded_from_file:
                    self._add_input_data_file(f)
        self.obs = obs
        self._check_num_groups(pars)
        self._e_ini = ini_merger["e0"]

        # some transformations we need
        self.transform_obs("theta", "theta_deg", lambda x: x*180/np.pi)
        # collapse observations
        self.collapse_observations(["theta", "theta_deg", "a", "e"])


    def set_stan_dict(self, num_outsamples):
        """
        Set the Stan data dictionary used for sampling

        Parameters
        ----------
        num_outsamples : int
            Number of predictive out-of-sample points to use
        """
        self.stan_data = dict(
            theta1 = self.obs_collapsed["theta"],
            theta_deg = self.obs_collapsed["theta_deg"],
            ecc = self.obs_collapsed["e"],
            N1 = self.num_obs,
            N2 = num_outsamples
        )

        self.stan_data = dict(
            theta2 = np.linspace(
                        min(self.stan_data["theta1"]),
                        max(self.stan_data["theta1"]),
                        self.stan_data["N2"]
            )
        )

        self.stan_data = dict(
            theta2_deg = self.stan_data["theta2"] * 180/np.pi
        )


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


    def all_plots(self, figsize=None, prior=False):
        """
        Plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        if prior:
            plot_func = self.prior_plot
            type_str = "prior"
        else:
            plot_func = self.posterior_plot
            type_str = "posterior"
        self.rename_dimensions({"eta_dim_0":"dim"})

        self.parameter_diagnostic_plots(self.latent_qtys, labeller=self._labeller_latent)

        # expand variables along a new dimension to match eta
        for k in ("alpha", "rho", "sigma"):
            self._fit_for_az[type_str][k] = self._fit_for_az[type_str][k].expand_dims({"dim":np.arange(self._fit_for_az[type_str].dims["dim"])}, axis=-1)

        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_xlabel(r"$\theta\degree$")
        ax.set_ylabel(r"$e_\mathrm{h}$")
        plot_func(xobs="theta_deg", yobs="e", xmodel="theta2_deg", ymodel="y", ax=ax)
        
        # latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"dim"})
        fig1 = ax1[0,0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{type_str}_{self._parameter_corner_plot_counter}"), fig=fig1)

        # marginal distribution of e_h
        self.plot_generated_quantity_dist(["y"], xlabels=[r"$e_\mathrm{h}$"])
