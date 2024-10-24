from abc import abstractmethod
import os.path
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from arviz import plot_kde
import dask
from datetime import datetime
from analysis.analysis_classes.HMQuantitiesBinaryData import HMQuantitiesBinaryData
from analysis.analysis_classes.StanModel import HierarchicalModel_2D
from analysis.analyse_ketju import determine_merger_timescale
from env_config import _cmlogger, date_format
from general import units
from plotting import savefig
from utils import get_files_in_dir


__all__ = ["_QuinlanModelBase", "QuinlanModelSimple", "QuinlanModelHierarchy"]

_logger = _cmlogger.getChild(__name__)


class _QuinlanModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        """
        Abstract class for Quinlan models. See HierarchicalModel_2D for parameters.
        """
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self._folded_qtys = ["inv_a", "ecc"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["HGp_s", "inv_a_0", "K", "e0", "a_err", "e_err"]
        self._latent_qtys_posterior = [
            f"{v}_posterior" if "_err" not in v else v for v in self.latent_qtys
        ]
        self._folded_qtys_labs = [r"$\mathrm{kpc}/a$", r"$e$"]
        self._latent_qtys_labs = [
            r"$H'(\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$",
            r"$\mathrm{kpc}/a_0$",
            r"$K$",
            r"$e_0$",
            r"$\mathrm{err}_{1/a}$",
            r"$\mathrm{err}_e$",
        ]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._labeller_latent_posterior = MapLabeller(
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs))
        )
        self._merger_id = None
        self._set_merger_time_file()

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

    @property
    def merger_time_file(self):
        return self._merger_time_file

    def _set_merger_time_file(self):
        if self._fit is None:
            self._merger_time_file = None
        else:
            d = os.path.dirname(self._fit.runset.csv_files[0])
            tstamp = self._get_timestamp_from_csv(self._fit.runset.csv_files[0])
            self._merger_time_file = os.path.join(d, f"merger_time-{tstamp}.csv")

    @abstractmethod
    def extract_data(self, pars, d=None):
        """
        Extract data from HMQcubes required for analysis

        Parameters
        ----------
        pars : dict
            analysis parameters
        d : path-like, optional
            HMQ data directory, by default None (paths read from
            `_input_data_files`)
        """
        self._set_merger_time_file()
        d = self._get_data_dir(d)
        obs = {"t": [], "a": [], "e": [], "e_ini": [], "m1": [], "m2": []}
        if self._loaded_from_file:
            fnames = d[0]
        else:
            fnames = get_files_in_dir(d)
            _logger.debug(f"Reading from dir: {d}")
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesBinaryData.load_from_file(f)
            status, idx0 = hmq.idx_finder(
                np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis
            )
            if not status:
                continue
            status, idx1 = hmq.idx_finder(
                pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis
            )
            if not status:
                continue
            try:
                assert idx0 < idx1
            except AssertionError:
                _logger.exception(
                    f"Lower index {idx0} (value: {np.nanmedian(hmq.hardening_radius):.3f}) is not less than upper index {idx1} (value: {pars['bh_binary']['target_semimajor_axis']['value']:.3f})!",
                    exc_info=True,
                )
                raise
            idxs = np.r_[idx0:idx1]
            obs["t"].append(hmq.binary_time[idxs])
            obs["a"].append(hmq.semimajor_axis[idxs])
            obs["e"].append(hmq.eccentricity[idxs])
            obs["m1"].append([hmq.binary_masses[0]])
            obs["m2"].append([hmq.binary_masses[1]])
            obs["e_ini"].append([hmq.initial_galaxy_orbit["e0"]])
            if self._merger_id is None:
                self._merger_id = re.sub("_[a-z]-", "-", hmq.merger_id)
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        self.obs = obs
        self._check_num_groups(pars)

        # some transformations we need
        self.transform_obs("a", "inva", lambda x: 1 / x)
        self.transform_obs("t", "t_shift", lambda x: x - x[0])
        self.collapse_observations(["t_shift", "inva", "e"])

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
        self.stan_data["t_OOS"] = np.linspace(
            np.max([t[0] for t in self.obs["t_shift"]]),
            np.min([t[-1] for t in self.obs["t_shift"]]),
            self.num_OOS,
        )

    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling
        """
        self.stan_data = dict(
            N_tot=self.num_obs,
            N_groups=self.num_groups,
            group_id=self.obs_collapsed["label"],
            t=self.obs_collapsed["t_shift"],
            e_ini=np.nanmedian([v for v in self.obs["e_ini"]]),
            inv_a=self.obs_collapsed["inva"],
            ecc=self.obs_collapsed["e"],
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
            self._determine_num_OOS("inv_a_posterior")
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
        fig, ax = plt.subplots(3, 2, figsize=figsize)
        try:
            self.plot_generated_quantity_dist(
                self.latent_qtys_posterior, ax=ax, xlabels=self._latent_qtys_labs
            )
        except ValueError:
            self.plot_generated_quantity_dist(
                self.latent_qtys, ax=ax, xlabels=self._latent_qtys_labs
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
        # rename dimensions for collapsing
        self.rename_dimensions(
            {
                "HGp_s_dim_0": "group",
                "inv_a_0_dim_0": "group",
                "K_dim_0": "group",
                "e0_dim_0": "group",
            }
        )
        self._expand_dimension(["a_err", "e_err"], "group")

        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex="all")
        ax[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        for axi, l in zip(ax, self._folded_qtys_labs):
            axi.set_ylabel(l)
        self.plot_predictive(
            xobs="t_shift",
            yobs="inva",
            xmodel="t",
            ymodel="inv_a_prior",
            ax=ax[0],
            save=False,
        )
        self.plot_predictive(
            xobs="t_shift",
            yobs="e",
            xmodel="t",
            ymodel="ecc_prior",
            ax=ax[1],
            show_legend=False,
        )

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=figsize,
            labeller=self._labeller_latent,
            combine_dims={"group"},
        )
        fig1 = ax1[0, 0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"
            ),
            fig=fig1,
        )

    @abstractmethod
    def all_posterior_pred_plots(self):
        pass

    def determine_merger_timescale_distribution(self, n=1000, save=True):
        """
        Deterministic calculation to sample merger timescale distribution.
        Latent parameters are sampled from the corresponding `_posterior`
        variable, which are replicated using the constrained hyperparameters.
        See https://mc-stan.org/docs/stan-users-guide/mixed-replication.html

        Parameters
        ----------
        n : int, optional
            number of samplings, by default 1000
        save : bool, optional
            save calculation to .csv file, by default True
        """
        self._set_merger_time_file()
        samples = dict.fromkeys(self.latent_qtys)
        for k in ("a_err", "e_err"):
            samples.pop(k)
        gq_suffix = "_posterior" if self._fit is not None else ""
        quantiles = self._rng.uniform(size=n)
        for k in samples.keys():
            samples[k] = np.nanquantile(
                self.sample_generated_quantity(f"{k}{gq_suffix}", state="OOS"),
                quantiles,
            )
        samples["a0"] = 1 / samples["inv_a_0"]
        samples.pop("inv_a_0")
        try:
            m1 = np.unique([m for m in self.obs["m1"]])
            m2 = np.unique([m for m in self.obs["m2"]])
            assert len(m1) == 1 and len(m2) == 1
            m1 = m1[0]
            m2 = m2[0]
        except AssertionError:
            _logger.exception(
                f"BH masses must be the same between runs, but have unique masses {m1} and {m2}",
                exc_info=True,
            )
            raise

        # deterministic calculation with Peter's formula
        # parallelise with dask
        start_time = datetime.now()
        _logger.info(
            f"Merger timescale calculation started: {start_time.strftime(date_format)}"
        )
        results = []
        tf = max([t[-1] for t in self.obs["t_shift"]])
        for i, d in enumerate(
            map(lambda *x: dict(zip(samples.keys(), x)), *samples.values())
        ):
            d.update({"t0": 0, "tf": tf, "m1": m1, "m2": m2})
            # unit conversions to Ketju units
            d["a0"] *= units.kpc
            d["tf"] *= units.Myr
            d["HGp_s"] *= 1e-9  # convert [Myr kpc]^-1 -> [yr pc]^-1
            results.append(dask.delayed(determine_merger_timescale)(**d))
        results = dask.compute(*results)
        self.merger_time = np.array(results)
        end_time = datetime.now()
        _logger.info(
            f"Merger timescale calculation ended: {end_time.strftime(date_format)}"
        )
        _logger.info(
            f"Merger timescale determined in {(end_time-start_time).total_seconds():.1f}s"
        )
        if save and self._fit is not None:
            # save merger time to .csv file so we don't have to always
            # recompute it
            np.savetxt(self.merger_time_file, self.merger_time, delimiter=",")

    def plot_merger_timescale(self, recalculate=False, figsize=None, **calc_kwargs):
        """
        Plot the merger timescale distribution.

        Parameters
        ----------
        recalculate : bool, optional
            re-evaluate the timescale if already existing, by default False
        """
        if (
            not recalculate
            and self.merger_time_file is not None
            and os.path.exists(self.merger_time_file)
        ):
            self.merger_time = np.loadtxt(self.merger_time_file, delimiter=",")
            _logger.info(f"Merger timescale read from file {self.merger_time_file}")
        else:
            self.determine_merger_timescale_distribution(**calc_kwargs)
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(r"$(t_\mathrm{merge} - t_\mathrm{h})/\mathrm{Myr}$")
        ax.set_ylabel(r"$\mathrm{PDF}$")
        plot_kde(self.merger_time, figsize=figsize, ax=ax)
        if ax.get_xlim()[0] < 0:
            ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        fig = ax.get_figure()
        savefig(self._make_fig_name(self.figname_base, "merger-timescale"), fig=fig)


class QuinlanModelSimple(_QuinlanModelBase):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        """
        Non-hierarchical Quinlan model. See HierarchicalModel_2D for parameters.
        """
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)

    def extract_data(self, pars, d=None):
        """
        See docs for `_QuinlanModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(
            self.figname_base,
            f"{self.merger_id}/quinlan-hardening-{self.merger_id}-simple",
        )

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
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent
        )

        # posterior predictive check
        fig1, ax1 = plt.subplots(2, 1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        for axi, l in zip(ax1, self._folded_qtys_labs):
            axi.set_ylabel(l)
        self.plot_predictive(
            xobs="t", yobs="inva", ymodel="inv_a_posterior", ax=ax1[0], save=False
        )
        self.plot_predictive(
            xobs="t", yobs="e", ymodel="ecc_posterior", ax=ax1[1], show_legend=False
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


class QuinlanModelHierarchy(_QuinlanModelBase):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        """
        Hierarchical Quinlan model. See HierarchicalModel_2D for parameters.
        """
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)
        self._hyper_qtys = [
            "HGp_s_mean",
            "HGp_s_std",
            "inv_a_0_mean",
            "inv_a_0_std",
            "K_mean",
            "K_std",
            "e0_mean",
            "e0_std",
        ]
        self._hyper_qtys_labs = [
            r"$\mu_{H'}$",
            r"$\sigma_{H'}$",
            r"$\mu_{1/a_0}$",
            r"$\sigma_{1/a_0}$",
            r"$\mu_K$",
            r"$\sigma_K$",
            r"$\mu_{e_0}$",
            r"$\sigma_{e_0}$",
        ]
        self._labeller_hyper = MapLabeller(
            dict(zip(self._hyper_qtys, self._hyper_qtys_labs))
        )
        self.merger_time = None

    def extract_data(self, pars, d=None):
        """
        See docs for `_QuinlanModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'hierarchy'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(
            self.figname_base,
            f"{self.merger_id}/quinlan-hardening-{self.merger_id}-hierarchy",
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
        self._prep_dims()
        self._expand_dimension(["a_err", "e_err"], "group")

        # hyperparameter plots
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive checks
        fig1, ax1 = plt.subplots(2, 1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        for axi, l in zip(ax1, self._folded_qtys_labs):
            axi.set_ylabel(l)
        self.plot_predictive(
            xobs="t_shift",
            yobs="inva",
            xmodel="t",
            ymodel="inv_a_posterior",
            ax=ax1[0],
            save=False,
        )
        self.plot_predictive(
            xobs="t_shift",
            yobs="e",
            xmodel="t",
            ymodel="ecc_posterior",
            ax=ax1[1],
            show_legend=False,
        )

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(
            self._latent_qtys_posterior,
            figsize=figsize,
            labeller=self._labeller_latent_posterior,
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

        # out of sample posterior
        fig1, ax1 = plt.subplots(2, 1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        for axi, l in zip(ax1, self._folded_qtys_labs):
            axi.set_ylabel(l)
        self.posterior_OOS_plot(
            xmodel="t_OOS", ymodel="inv_a_posterior", ax=ax1[0], save=False
        )
        self.posterior_OOS_plot(
            xmodel="t_OOS", ymodel="ecc_posterior", ax=ax1[1], show_legend=False
        )
