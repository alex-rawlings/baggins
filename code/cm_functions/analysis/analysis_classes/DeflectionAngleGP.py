import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import HierarchicalModel_2D, HMQuantitiesBinaryData
from .. import first_major_deflection_angle, find_idxs_of_n_periods
from ...env_config import _cmlogger
from ...plotting import savefig
from ...utils import get_files_in_dir


__all__ = ["DeflectionAngleGP"]

_logger = _cmlogger.getChild(__name__)


class DeflectionAngleGP(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        """
        See HierarchicalModel_2D for parameters.
        """
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self._latent_qtys = ["rho", "alpha", "sigma", "eta"]
        self._latent_qtys_labs = [r"$\rho$", r"$\alpha$", r"$\sigma$", r"$\eta$"]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
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
        dirs : list, optional
            HMQ data directories to include, by default None (paths read from
            `_input_data_files`)
        tol : float
            relative tolerance for merger orbit consistency, by default 1e-5
        """
        dirs = self._get_data_dir(dirs)
        obs = {"theta": [], "a": [], "e": []}
        # dummy variables for consistency checks between directories
        ini_merger = {}
        for d in dirs:
            if self._loaded_from_file:
                fnames = d
            else:
                fnames = get_files_in_dir(d)
            for f in fnames:
                _logger.info(f"Loading file: {f}")
                hmq = HMQuantitiesBinaryData.load_from_file(f)
                # protect against instances where no data for bound binary
                try:
                    hmq.hardening_radius
                except AttributeError:
                    _logger.warning(
                        f"No attribute `hardening_radius` for file {f}: skipping."
                    )
                    continue
                theta = first_major_deflection_angle(hmq.prebound_deflection_angles)[0]
                if np.isnan(theta):
                    _logger.warning(
                        f"No prebound deflection angles in file {f}: skipping."
                    )
                    continue
                a = np.nanmedian(hmq.hardening_radius)
                status, hard_idx = hmq.idx_finder(a, hmq.semimajor_axis)
                if not status:
                    _logger.warning(
                        f"No semimajor axis values correspond to hardening radius in file {f}: skipping."
                    )
                    continue
                t_target = hmq.binary_time[hard_idx]
                _logger.debug(f"Target time: {t_target} Myr")
                try:
                    target_idx, delta_idxs = find_idxs_of_n_periods(
                        t_target,
                        hmq.binary_time,
                        hmq.binary_separation,
                        num_periods=pars["bh_binary"]["num_orbital_periods"],
                    )
                except IndexError:
                    _logger.warning(
                        f"Orbital period for hard semimajor axis not found! This run will not form part of the analysis."
                    )
                    continue
                _logger.debug(
                    f"For observation {self._input_data_file_count} found target time between indices {delta_idxs[0]} and {delta_idxs[1]}"
                )
                period_idxs = np.r_[delta_idxs[0] : delta_idxs[1]]
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
                            assert np.abs(hmq.initial_galaxy_orbit[k] - v) / v < tol
                        except AssertionError:
                            _logger.exception(
                                f"File {f} has an inconsistent merger orbit! Value of {k} is {hmq.initial_galaxy_orbit[k]}, should be {v}!",
                                exc_info=True,
                            )
                            raise
                if not self._loaded_from_file:
                    self._add_input_data_file(f)
        self.obs = obs
        self._check_num_groups(pars)
        self._e_ini = ini_merger["e0"]

        # some transformations we need
        self.transform_obs("theta", "theta_deg", lambda x: x * 180 / np.pi)
        # collapse observations
        self.collapse_observations(["theta", "theta_deg", "a", "e"])

    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables
        """
        try:
            assert self.num_OOS is not None
        except AssertionError:
            _logger.exception(
                f"num_OOS cannot be None when setting Stan data!", exc_info=True
            )
            raise
        self.stan_data = dict(
            N2=self.num_OOS,
            theta2=np.linspace(
                min(self.stan_data["theta1"]),
                max(self.stan_data["theta1"]),
                self.num_OOS,
            ),
        )
        self.stan_data = dict(theta2_deg=self.stan_data["theta2"] * 180 / np.pi)

    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling
        """
        self.stan_data = dict(
            theta1=self.obs_collapsed["theta"],
            theta_deg=self.obs_collapsed["theta_deg"],
            ecc=self.obs_collapsed["e"],
            N1=self.num_obs,
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()

    def sample_model(self, sample_kwargs=...):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS
        from previous sample.
        """
        super().sample_model(sample_kwargs)
        if self._loaded_from_file:
            self._determine_num_OOS("y")
            self._set_stan_data_OOS()

    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        return super().sample_generated_quantity(gq, force_resample, state)

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
        self.plot_generated_quantity_dist(
            self._latent_qtys, ax=ax, xlabels=self._latent_qtys_labs
        )
        return ax

    def all_plots(self, figsize=None):
        """
        Plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        type_str = "prior" if self._fit is None else "posterior"
        self.rename_dimensions({"eta_dim_0": "dim"})

        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent
        )

        # expand variables along a new dimension to match eta
        self._expand_dimension(["alpha", "rho", "sigma"], "dim")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$\theta\degree$")
        ax.set_ylabel(r"$e_\mathrm{h}$")
        self.plot_predictive(
            xmodel="theta2_deg", ymodel="y", xobs="theta_deg", yobs="e", ax=ax
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

        # marginal distribution of e_h
        self.plot_generated_quantity_dist(["y"], xlabels=[r"$e_\mathrm{h}$"])
