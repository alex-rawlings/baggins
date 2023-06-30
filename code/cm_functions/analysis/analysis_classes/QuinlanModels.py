import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from arviz import plot_kde
import dask
from datetime import datetime
from . import StanModel_2D, HMQuantitiesData
from ..orbit import determine_merger_timescale
from ...env_config import _cmlogger, date_format
from ...plotting import savefig
from ...utils import get_files_in_dir


__all__ = ["QuinlanModelSimple", "QuinlanModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _QuinlanModelBase(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["HGp_s", "inv_a_0", "K", "e0"]
        self._latent_qtys_posterior = [f"{v}_posterior" for v in self.latent_qtys]
        self._latent_qtys_labs = [r"$H'(\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$", r"$\mathrm{kpc}/a_0$", r"$K$", r"$e_0$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
        self._labeller_latent_posterior = MapLabeller(dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs)))
        self._merger_id = None
        self._gq_state = "pred"

    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    @property
    def merger_id(self):
        return self._merger_id


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
        d = self._get_data_dir(d)
        obs = {"t":[], "a":[], "e":[], "e_ini":[], "m1":[], "m2":[]}
        if self._loaded_from_file:
            fnames = d[0]
        else:
            fnames = get_files_in_dir(d)
            _logger.logger.debug(f"Reading from dir: {d}")
        for f in fnames:
            _logger.logger.info(f"Loading file: {f}")
            hmq = HMQuantitiesData.load_from_file(f)
            status, idx0 = hmq.idx_finder(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
            if not status: continue
            status, idx1 = hmq.idx_finder(pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis)
            if not status: continue
            try:
                assert idx0 < idx1
            except AssertionError:
                _logger.logger.exception(f"Lower index {idx0} (value: {np.nanmedian(hmq.hardening_radius):.3f}) is not less than upper index {idx1} (value: {pars['bh_binary']['target_semimajor_axis']['value']:.3f})!", exc_info=True)
                raise
            idxs = np.r_[idx0:idx1]
            obs["t"].append(hmq.binary_time[idxs])
            obs["a"].append(hmq.semimajor_axis[idxs])
            obs["e"].append(hmq.eccentricity[idxs])
            obs["m1"].append([hmq.binary_masses[0]])
            obs["m2"].append([hmq.binary_masses[1]])
            obs["e_ini"].append([hmq.initial_galaxy_orbit["e0"]])
            if self._merger_id is None:
                # TODO may not work if sample is not MC generated
                _id = hmq.merger_id.split("-")[:2]
                self._merger_id = "-".join([ii[:-2] for ii in _id])

            if not self._loaded_from_file:
                self._add_input_data_file(f)
        if not obs["e_ini"]:
            obs.pop("e_ini")
        self.obs = obs
        self._check_num_groups(pars)

        # some transformations we need
        self.transform_obs("a", "inva", lambda x:1/x)
        self.transform_obs("t", "t_shift", lambda x:x-x[0])
        self.collapse_observations(["t_shift", "inva", "e"])


    def set_stan_dict(self, N_OOS=None):
        """
        Set the Stan data dictionary used for sampling
        """
        self.stan_data = dict(
            N_tot = self.num_obs,
            N_groups = self.num_groups,
            group_id = self.obs_collapsed["label"],
            t = self.obs_collapsed["t_shift"],
            e_ini = np.nanmedian([v for v in self.obs["e_ini"]]),
            inv_a = self.obs_collapsed["inva"],
            ecc = self.obs_collapsed["e"]
        )
        if N_OOS is None:
            try:
                assert self._gq_state == "pred"
            except AssertionError:
                _logger.logger.exception(f"Cannot reset stan data to predictive state", exc_info=True)
                raise
            self.stan_data["N_OOS"] = self.stan_data["N_tot"]
            self.stan_data["t_OOS"] = self.stan_data["t"]
        else:
            self.stan_data["N_OOS"] = N_OOS
            self.stan_data["t_OOS"] = np.linspace(
                                            np.min(self.stan_data["t"]),
                                            np.max(self.stan_data["t"]),
                                            N_OOS
            )
            self._gq_state ="OOS"



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
        self.plot_generated_quantity_dist(self.latent_qtys_posterior, ax=ax, xlabels=self._latent_qtys_labs)
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
        samples = dict.fromkeys(self.latent_qtys)
        quantiles = self._rng.uniform(size=n)
        for k in samples.keys():
            samples[k] = np.nanquantile(self.sample_generated_quantity(f"{k}_posterior"), quantiles)
        samples["a0"] = 1/samples["inv_a_0"]
        samples.pop("inv_a_0")
        try:
            m1 = np.unique([m for m in self.obs["m1"]])
            m2 = np.unique([m for m in self.obs["m2"]])
            assert len(m1)==1 and len(m2)==1
            m1 = m1[0]
            m2 = m2[0]
        except AssertionError:
            _logger.logger.exception(f"BH masses must be the same between runs, but have unique masses {m1} and {m2}", exc_info=True)
            raise

        # deterministic calculation with Peter's formula
        # parallelise with dask
        start_time = datetime.now()
        _logger.logger.info(f"Merger timescale calculation started: {start_time.strftime(date_format)}")
        results = []
        tf = max([t[-1] for t in self.obs["t_shift"]])
        for d in map(lambda *x: dict(zip(samples.keys(), x)), *samples.values()):
            d.update({"t0":0, "tf":tf, "m1":m1, "m2":m2})
            results.append(
                dask.delayed(determine_merger_timescale)(**d)
            )
        results = dask.compute(*results)
        self.merger_time = np.full(n, np.nan)
        for i, r in enumerate(results):
            self.merger_time[i] = r[0]
        end_time = datetime.now()
        _logger.logger.info(f"Merger timescale calculation ended: {end_time.strftime(date_format)}")
        _logger.logger.info(f"Merger timescale determined in {(end_time-start_time).total_seconds():.1f}s")
        if save:
            # save merger time to .csv file so we don't have to always 
            # recompute it
            np.savetext(self.merger_time_file, self.merger_time, delimiter=",")


    def plot_merger_timescale(self, recalculate=False, figsize=None, **calc_kwargs):
        """
        Plot the merger timescale distribution.

        Parameters
        ----------
        recalculate : bool, optional
            re-evaluate the timescale if already existing, by default False
        """
        if not recalculate and self.merger_time_file is not None and os.path.exists(self.merger_time_file):
            self.merger_time = np.loadtxt(self.merger_time_file, delimiter=",")
        else:
            self.determine_merger_timescale_distribution(**calc_kwargs)
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel(r"$t_\mathrm{merge}$")
        ax.set_ylabel(r"$\mathrm{PDF}$")
        plot_kde(self.merger_time, figsize=figsize, ax=ax)
        fig = ax.get_figure()
        savefig(self._make_fig_name(self.figname_base, f"merger-timescale"), fig=fig)




class QuinlanModelSimple(_QuinlanModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)


    def extract_data(self, pars, d=None):
        """
        See docs for `_QuinlanModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/quinlan-hardening-{self.merger_id}-simple")


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
        self.parameter_diagnostic_plots(self.latent_qtys, labeller=self._labeller_latent)

        # posterior predictive check
        fig1, ax1 = plt.subplots(2,1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        ax1[0].set_ylabel(r"$mathrm{kpc}/a$")
        ax1[1].set_ylabel(r"$e$")
        self.posterior_pred_plot(xobs="t", yobs="inva", ymodel="inv_a_posterior", ax=ax1[0], save=False)
        self.posterior_pred_plot(xobs="t", yobs="e", ymodel="ecc_posterior", ax=ax1[1], show_legend=False)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)
        
        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax




class QuinlanModelHierarchy(_QuinlanModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._hyper_qtys = ["HGp_s_mean", "HGp_s_std", "inv_a_0_mean", "inv_a_0_std", "K_mean", "K_std", "e0_mean", "e0_std"]
        self._hyper_qtys_labs = [r"$\mu_{H'}$", r"$\sigma_{H'}$", r"$\mu_{1/a_0}$", r"$\sigma_{1/a_0}$", r"$\mu_K$", r"$\sigma_K$", r"$\mu_{e_0}$", r"$\sigma_{e_0}$"]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qtys_labs)))
        self.merger_time = None
        self._set_merger_time_file()


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


    def extract_data(self, pars, d=None):
        """
        See docs for `_QuinlanModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'hierarchy'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/quinlan-hardening-{self.merger_id}-hierarchy")


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
        try:
            assert self._gq_state == "pred"
        except AssertionError:
            _logger.logger.exception(f"Stan data is not in 'predictive' state: generated quantities will be computed for out-of-sample (OOS) values! Must run predictive checks before doing OOS calculations.")
        self._prep_dims()

        # hyperparameter plots
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive checks
        fig1, ax1 = plt.subplots(2,1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        ax1[0].set_ylabel(r"$\mathrm{kpc}/a$")
        ax1[1].set_ylabel(r"$e$")
        self.posterior_pred_plot(xobs="t_shift", yobs="inva", xmodel="t", ymodel="inv_a_posterior", ax=ax1[0], save=False)
        self.posterior_pred_plot(xobs="t_shift", yobs="e", xmodel="t", ymodel="ecc_posterior", ax=ax1[1], show_legend=False)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(self._latent_qtys_posterior, figsize=figsize, labeller=self._labeller_latent_posterior, combine_dims={"group"})
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax


    def all_posterior_OOS_plots(self, N, figsize=None):
        self.set_stan_dict(N)
        # force resampling of generated quantities, but we don't need the 
        # return value here
        self.sample_generated_quantity(self.latent_qtys_posterior[0], force_resample=True)
        self._prep_dims()

        # out of sample posterior
        fig1, ax1 = plt.subplots(2,1, figsize=figsize, sharex="all")
        ax1[1].set_xlabel(r"$t'/\mathrm{Myr}$")
        ax1[0].set_ylabel(r"$\mathrm{kpc}/a$")
        ax1[1].set_ylabel(r"$e$")
        self.posterior_OOS_plot(xmodel="t_OOS", ymodel="inv_a_posterior", ax=ax1[0], save=False)
        self.posterior_OOS_plot(xmodel="t_OOS", ymodel="ecc_posterior", ax=ax1[1], show_legend=False)
