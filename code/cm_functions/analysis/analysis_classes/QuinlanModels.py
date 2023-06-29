import os.path
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
import dask
from datetime import datetime
from . import StanModel_2D, HMQuantitiesData
from ..orbit import analytic_evolve_peters_quinlan
from ...env_config import _cmlogger
from ...plotting import savefig
from ...utils import get_files_in_dir


__all__ = ["QuinlanModelSimple", "QuinlanModelHierarchy"]

_logger = _cmlogger.copy(__file__)


class _QuinlanModelBase(StanModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["HGp_s", "inv_a_0", "K", "e0"]
        self._latent_qtys_labs = [r"$H'(\mathrm{kpc}^{-1} \mathrm{Myr}^{-1})$", r"$\mathrm{kpc}/a_0$", r"$K$", r"$e_0$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
        self._merger_id = None

    @property
    def latent_qtys(self):
        return self._latent_qtys

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
                self._merger_id = hmq.merger_id

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


    def set_stan_dict(self):
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
        self._hyper_qtys = ["HGp_s_mean", "HGp_s_std", "inv_a_0_mean", "inv_a_0_std", "K_mean", "K_std", "e0_mean", "e0_std"]
        self._hyper_qtys_labs = [r"$\mu_{H'}$", r"$\sigma_{H'}$", r"$\mu_{1/a_0}$", r"$\sigma_{1/a_0}$", r"$\mu_K$", r"$\sigma_K$", r"$\mu_{e_0}$", r"$\sigma_{e_0}$"]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qtys_labs)))


    def extract_data(self, pars, d=None):
        """
        See docs for `_QuinlanModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'hierarchy'
        """
        super().extract_data(pars, d)
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/quinlan-hardening-{self.merger_id}-hierarchy")


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


    def determine_merger_timescale_distribution(self, n=1000):
        """
        Deterministic calculation to sample merger timescale distribution

        Parameters
        ----------
        n : int, optional
            number of samplings, by default 1000

        Returns
        -------
        merger_time : np.ndarray
            sampled values of merger timescale
        """
        # TODO need to sample hyperparameters, not latent parameters!
        samples = dict.fromkeys(self.latent_qtys)
        quantiles = self._rng.uniform(size=n)
        for k in samples.keys():
            samples[k] = np.nanquantile(self.sample_generated_quantity(k), quantiles)
        samples["a0"] = 1/samples["inv_a_0"]
        samples.pop("inv_a_0")
        try:
            m1 = np.unique([m for m in self.obs["m1"]])
            m2 = np.unique([m for m in self.obs["m2"]])
            assert len(m1)==1 and len(m2)==2
        except AssertionError:
            _logger.logger.exception(f"BH masses must be the same between runs, but have unique masses {m1} and {m2}", exc_info=True)
            raise

        # deterministic calculation with Peter's formula
        # parallelise with dask
        start_time = datetime.now()
        results = []
        tf = max([t[-1] for t in self.obs["t_shift"]])
        for d in map(lambda *x: dict(zip(samples.keys(), x))):
            d.update({"t0":0, "tf":tf, "m1":m1, "m2":m2})
            results.append(
                dask.delayed(analytic_evolve_peters_quinlan)(**d)
            )
        results = dask.compute(*results)
        merger_time = np.full(n, np.nan)
        for i, r in enumerate(results):
            merger_time[i] = r[0]
        _logger.logger.info(f"Merger timescale determined in {datetime.now()-start_time:.1f}s")
        return merger_time