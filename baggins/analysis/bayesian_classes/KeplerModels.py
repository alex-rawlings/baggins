from abc import abstractmethod
import os.path
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_1D
from baggins.analysis.data_classes.HMQuantitiesBinaryData import (
    HMQuantitiesBinaryData,
)
from baggins.analysis.analyse_ketju import find_idxs_of_n_periods
from baggins.env_config import _cmlogger
from baggins.plotting import savefig
from ketjugw.units import unit_length_in_pc, unit_time_in_years

__all__ = ["_KeplerModelBase", "KeplerModelSimple", "KeplerModelHierarchy"]

_logger = _cmlogger.getChild(__name__)


class _KeplerModelBase(HierarchicalModel_1D):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self._folded_qtys = ["log10_angmom", "log10_energy"]
        self._folded_qtys_labs = [
            r"$\log_{10}\left( \left(l/\sqrt{GM}\right)/\sqrt{\mathrm{pc}} \right)$",
            r"$\log_{10}\left(\left(|E|/\left(GM_1M_2 \right)\right)/\left( \mathrm{M}_\odot \mathrm{pc}^2\mathrm{yr}^{-2} \right) \right)$",
        ]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["a_hard", "e_hard", "a_err", "e_err"]
        self._latent_qtys_labs = [r"$a_\mathrm{h}/\mathrm{pc}$", "$e_\mathrm{h}$"]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._latent_qtys_posterior = [
            f"{v}_posterior" if "_err" not in v else v for v in self.latent_qtys
        ]
        self._labeller_latent_posterior = MapLabeller(
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs))
        )
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
    def merger_id(self):
        return self._merger_id

    @abstractmethod
    def extract_data(self, dir, pars):
        """
        Extract data and manipulations required for the Kepler binary model

        Parameters
        ----------
        dir : list
            list of paths to hierarchical model data files
        pars : dict
            analysis parameters
        """
        obs = {
            "angmom": [],
            "energy": [],
            "a": [],
            "e": [],
            "mass1": [],
            "mass2": [],
            "star_mass": [],
            "e_ini": [],
            "t": [],
        }
        for i, f in enumerate(dir):
            _logger.debug(f"Loading file: {f}")
            hmq = HMQuantitiesBinaryData.load_from_file(f)
            status, idx = hmq.idx_finder(
                np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis
            )
            if not status:
                continue
            t_target = hmq.binary_time[idx]
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
                    "Orbital period for hard semimajor axis not found! This run will not form part of the analysis."
                )
                continue
            _logger.debug(
                f"For observation {i} found target time between indices {delta_idxs[0]} and {delta_idxs[1]}"
            )
            period_idxs = np.r_[delta_idxs[0] : delta_idxs[1]]
            obs["angmom"].append(hmq.binary_angular_momentum[period_idxs])
            obs["energy"].append(-hmq.binary_energy[period_idxs])
            # convert semimajor axis from kpc to pc here
            obs["a"].append(hmq.semimajor_axis[period_idxs] * 1000)
            obs["e"].append(hmq.eccentricity[period_idxs])
            obs["t"].append(hmq.binary_time[period_idxs])
            try:
                obs["mass1"].append([hmq.binary_masses[0]])
                obs["mass2"].append([hmq.binary_masses[1]])
            except AttributeError:
                _logger.error(
                    f"Attribute 'particle_masses' does not exist for {f}! Will assume equal-mass BHs!"
                )
                obs["mass1"].append([hmq.masses_in_galaxy_radius["bh"][0] / 2])
                obs["mass2"].append([hmq.masses_in_galaxy_radius["bh"][0] / 2])
            obs["e_ini"].append([hmq.initial_galaxy_orbit["e0"]])
            obs["star_mass"].append([hmq.particle_masses["stars"]])
            if self._merger_id is None:
                self._merger_id = re.sub("_[a-z]-", "-", hmq.merger_id)
        self.obs = obs
        self._check_num_groups(pars)

        # some transformations we need
        # G in correct units
        G_in_Msun_pc_yr = unit_length_in_pc**3 / unit_time_in_years**2
        # mass transforms
        self.transform_obs(("mass1", "mass2"), "total_mass", lambda x, y: x + y)
        self.transform_obs(("mass1", "mass2"), "mass_product", lambda x, y: x * y)
        self.obs["total_mass_long"] = []
        self.obs["mass_product_long"] = []
        for tm, mp, am in zip(
            self.obs["total_mass"], self.obs["mass_product"], self.obs["angmom"]
        ):
            self.obs["total_mass_long"].append(np.repeat(tm, len(am)))
            self.obs["mass_product_long"].append(np.repeat(mp, len(am)))
        self.transform_obs(
            "total_mass_long", "log10_total_mass_long", lambda x: np.log10(x)
        )
        # angular momentum transformations
        self.transform_obs(
            "angmom",
            "angmom_corr",
            lambda x: x * unit_length_in_pc**2 / unit_time_in_years,
        )
        self.transform_obs(
            ("angmom_corr", "total_mass_long"),
            "angmom_corr_red",
            lambda j, m: j / np.sqrt(G_in_Msun_pc_yr * m),
        )
        self.transform_obs(
            "angmom_corr_red", "log10_angmom_corr_red", lambda x: np.log10(x)
        )
        # energy transformations
        self.transform_obs(
            "energy",
            "energy_corr",
            lambda x: x * unit_length_in_pc**2 / unit_time_in_years**2,
        )
        self.transform_obs(
            ("energy_corr", "mass_product_long"),
            "energy_corr_red",
            lambda e, m: e / (G_in_Msun_pc_yr * m),
        )
        self.transform_obs(
            "energy_corr_red", "log10_energy_corr_red", lambda x: np.log10(x)
        )
        # collapse observations
        self.collapse_observations(
            [
                "log10_angmom_corr_red",
                "log10_energy_corr_red",
                "a",
                "e",
                "total_mass_long",
                "log10_total_mass_long",
            ]
        )

    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables
        """
        try:
            assert self.num_OOS is not None
        except AssertionError:
            _logger.exception(
                "num_OOS cannot be None when setting Stan data!", exc_info=True
            )
            raise
        self.stan_data["N_OOS"] = self.num_OOS
        self.stan_data["group_id_OOS"] = self._rng.integers(
            1, self.num_groups, size=self.num_OOS, endpoint=True
        )
        # TODO add out of sample quantities

    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling
        """
        self.stan_data = dict(
            N_obs=self.num_obs,
            N_groups=self.num_groups,
            group_id=self.obs_collapsed["label"],
            e0=self.obs["e_ini"],
            log10_angmom=self.obs_collapsed["log10_angmom_corr_red"],
            log10_energy=self.obs_collapsed["log10_energy_corr_red"],
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()

    def sample_model(self, sample_kwargs={}):
        super().sample_model(sample_kwargs)
        if self._loaded_from_file:
            self._determine_num_OOS("log10_angmom_posterior")
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
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        self.plot_generated_quantity_dist(
            self._latent_qtys, ax=ax, xlabels=self._latent_qtys_labs
        )
        return ax

    def all_prior_plots(self, figsize=None):
        """
        Prior plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(
            r"$\log\left( \left(l/\sqrt{GM}\right)/\sqrt{\mathrm{pc}} \right)$"
        )
        ax.set_ylabel("PDF")
        self.prior_plot(xobs="log10_angmom_corr_red", xmodel="log10_angmom", ax=ax)

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)


class KeplerModelSimple(_KeplerModelBase):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self.figname_base = f"{self.figname_base}-simple"

    def extract_data(self, pars, d=None):
        """
        See docs for `_KeplerModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(
            self.figname_base,
            f"{self.merger_id}/binary_properties-{self.merger_id}-simple",
        )

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
        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent
        )

        # posterior predictive check
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax1.set_xlabel(
            r"$\log\left( \left(l/\sqrt{GM}\right)/\sqrt{\mathrm{pc}} \right)$"
        )
        ax1.set_ylabel("PDF")
        self.posterior_plot(
            xobs="log10_angmom_corr_red", xmodel="log10_angmom_posterior", ax=ax1
        )

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(
            self.latent_qtys, figsize=figsize, labeller=self._labeller_latent
        )
        fig = ax.flatten()[0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_{self._parameter_corner_plot_counter}"
            ),
            fig=fig,
        )
        return ax


class KeplerModelHierarchy(_KeplerModelBase):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self.figname_base = f"{self.figname_base}-hierarchy"
        self._hyper_qtys = ["a_hard_mu", "a_hard_sigma", "e_hard_mu", "e_hard_sigma"]
        self._hyper_qtys_labs = [
            r"$\mu_{a_\mathrm{h}}$",
            r"$\sigma_{a_\mathrm{h}}$",
            r"$\mu_{e_\mathrm{h}}$",
            r"$\sigma_{e_\mathrm{h}}$",
        ]
        self._labeller_hyper = MapLabeller(
            dict(zip(self._hyper_qtys, self._hyper_qtys_labs))
        )

    def extract_data(self, pars, d=None):
        """
        See docs for `_KeplerModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'hierarchy'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(
            self.figname_base,
            f"{self.merger_id}/binary_properties-{self.merger_id}-hierarchy",
        )

    def _prep_dims(self):
        """
        Rename dimensions for collapsing.
        """
        _rename_dict = {}
        for k in itertools.chain(self.latent_qtys, self._latent_qtys_posterior):
            _rename_dict[f"{k}_dim_0"] = "group"
        self.rename_dimensions(_rename_dict)

    def all_posterior_pred_plots(self, figsize=None):
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
        self._prep_dims()
        self._expand_dimension(["a_err", "e_err"], "group")

        # hyper parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive checks
        fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
        for axi, lab in zip(ax1, self._folded_qtys_labs):
            axi.set_xlabel(lab)
            axi.set_ylabel("PDF")
        self.plot_predictive(
            xobs="log10_angmom_corr_red",
            xmodel="log10_angmom_posterior",
            ax=ax1[0],
            save=False,
        )
        self.plot_predictive(
            xobs="log10_energy_corr_red", xmodel="log10_energy_posterior", ax=ax1[1]
        )

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        # TODO append a boundary value of e to data? Ensures full domain seen
        ax = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=figsize,
            labeller=self._labeller_latent,
            combine_dims={"group"},
        )
        fig = ax.flatten()[0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_{self._parameter_corner_plot_counter}"
            ),
            fig=fig,
        )
        return ax

    def all_posterior_OOS_plots(self, figsize=None):
        self._prep_dims()
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        for axi, lab in zip(ax, self._folded_qtys_labs):
            axi.set_xlabel(lab)
            axi.set_ylabel("PDF")
        for i, (q, s) in enumerate(zip(self._folded_qtys_posterior, (False, True))):
            self.posterior_OOS_plot(xmodel=q, ax=ax[i], save=s)
