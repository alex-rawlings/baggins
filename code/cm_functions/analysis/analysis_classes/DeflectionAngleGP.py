import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import StanModel_2D, HMQuantitiesData
from .. import first_major_deflection_angle, find_idxs_of_n_periods
from ...env_config import _cmlogger

__all__ = []

_logger = _cmlogger.copy(__file__)



class DeflectionAngleGP(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["rho", "alpha", "sigma", "eta"]
        self._latent_qtys_labs = [r"$\rho$", r"$\alpha$", r"$\sigma$", r"$\eta$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))


    @property
    def latent_qtys(self):
        return self._latent_qtys


    def extract_data(self, dirs, pars):
        """
        Extract data from HMQcubes required for analysis

        Parameters
        ----------
        dirs : list
            list of HMQ data directories to include
        pars : dict
            analysis parameters
        """
        obs = {"theta":[], "a":[], "e":[]}
        i = 0
        for d in dirs:
            for f in dir:
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
                _logger.logger.debug(f"For observation {i} found target time between indices {delta_idxs[0]} and {delta_idxs[1]}")
                period_idxs = np.r_[delta_idxs[0]:delta_idxs[1]]
                obs["theta"].append([theta])
                obs["a"].append([a])
                obs["e"].append([np.nanmedian(hmq.eccentricity[period_idxs])])
                i += 1
        try:
            assert i >= pars["stan"]["min_num_samples"]
        except AssertionError:
            _logger.logger.exception(f"We require no less than {pars['stan']['min_num_samples']} samples groups, but have only {i}!", exc_info=True)
            raise
        self.obs = obs

        # some transformations we need
        self.transform_obs("theta", "theta_deg", lambda x: x*180/np.pi)
        # collapse observations
        self.collapse_observations(["theta", "theta_deg", "a", "e"])





