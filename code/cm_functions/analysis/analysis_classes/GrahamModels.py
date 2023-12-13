from abc import abstractmethod
import os.path
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import HierarchicalModel_2D, FactorModel_2D
from . import HMQuantitiesBinaryData, HMQuantitiesSingleData
from ...mathematics import get_histogram_bin_centres
from ...env_config import _cmlogger
from ...plotting import savefig
from ...utils import get_files_in_dir

__all__ = ["GrahamModelSimple", "GrahamModelHierarchy"]

_logger = _cmlogger.getChild(__name__)


class _GrahamModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._folded_qtys = ["log10_surf_rho"]
        self._folded_qtys_labs = [r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["rb", "Re", "log10densb", "g", "n", "a"]
        self._latent_qtys_posterior = [f"{v}_posterior" for v in self.latent_qtys]
        self._latent_qtys_labs = [r"$r_\mathrm{b}/\mathrm{kpc}$", r"$R_\mathrm{e}/\mathrm{kpc}$", r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$", r"$\gamma$", r"$n$", r"$a$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))
        self._labeller_latent_posterior = MapLabeller(dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs)))
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
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    @property
    def merger_id(self):
        return self._merger_id


    @abstractmethod
    def extract_data(self, pars, d=None, binary=True):
        """
        Data extraction and manipulation required for the Graham density model

        Parameters
        ----------
        pars : dict
            analysis parameters
        d : path-like, optional
            HMQ data directory, by default None (paths read from 
            `_input_data_files`)
        binary: bool, optional
            system before merger (2 BHs present), by default True
        """
        obs = {"R":[], "proj_density":[], "vkick":[]}
        d = self._get_data_dir(d)
        if self._loaded_from_file:
            fnames = d[0]
        elif os.path.isfile(d):
            fnames = [d]
        else:
            fnames = get_files_in_dir(d)
            _logger.debug(f"Reading from dir: {d}")
        is_single_file = len(fnames)==1
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            if binary:
                _logger.debug("Hierarchical model will be constructed for a binary BH system")
                hmq = HMQuantitiesBinaryData.load_from_file(f)
                status, idx = hmq.idx_finder(pars["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis)
            else:
                _logger.debug("Hierarchical model will be constructed for a single BH system")
                hmq = HMQuantitiesSingleData.load_from_file(f)
                idx = 0
                status = True
            if not status: continue
            r = get_histogram_bin_centres(hmq.radial_edges)
            obs["R"].append(r)
            obs["proj_density"].append(list(hmq.projected_mass_density.values())[idx])
            # get median escape velocity within some radius
            vesc = np.nanmedian(list(hmq.escape_velocity.values())[idx][hmq.radial_edges < 1])
            obs["vkick"].append([hmq.merger_remnant['kick']/vesc])
            if self._merger_id is None:
                self._merger_id = re.sub("_[a-z]-", "-", hmq.merger_id)
                if not is_single_file:
                    # remove vXXXX from merger ID
                    self._merger_id = re.sub("-v[0-9]*", "", self._merger_id)
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        if is_single_file:
            # we have loaded a single file
            # manipulate the data so it "looks" like multiple files
            _obs = obs.copy()
            obs = {"R":[], "proj_density":[], "vkick":[]}
            for i in range(_obs["proj_density"][0].shape[0]):
                obs["R"].append(_obs["R"][0])
                obs["proj_density"].append(_obs["proj_density"][0][i,:])
                obs["vkick"].append(_obs["vkick"][0])
            _logger.warning("Observations from a single file have been converted to a hierarchy format")
        self.obs = obs
        # some transformations we need
        self.transform_obs("R", "log10_R", lambda x: np.log10(x))
        self.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))


    @abstractmethod
    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables. 
        Each derived class will need its own implementation, however all will
        require knowledge of the minimum and maximum radius to model: let's 
        do that here.
        """
        rmin = np.max([R[0] for R in self.obs["R"]])
        rmax = np.min([R[-1] for R in self.obs["R"]])
        return rmin, rmax


    @abstractmethod
    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling.
        """
        self.stan_data = dict(
            N_tot = self.num_obs_collapsed,
            N_groups = self.num_groups,
            group_idx = self.obs_collapsed["label"],
            R = self.obs_collapsed["R"],
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
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()


    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        v = super().sample_generated_quantity(gq, force_resample, state)
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state)
            return v[...,idxs]
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
        ncol = int(np.ceil(len(self.latent_qtys)/2))
        fig, ax = plt.subplots(2,ncol, figsize=figsize)
        try:
            self.plot_generated_quantity_dist(self.latent_qtys_posterior, ax=ax, xlabels=self._latent_qtys_labs)
        except:
            _logger.warning(f"Cannot plot latent distributions for `latent_qtys_posterior`, trying for `latent_qtys`.")
            self.plot_generated_quantity_dist(self.latent_qtys, ax=ax, xlabels=self._latent_qtys_labs)
        ax[1,0].set_xscale("log")
        return ax


    @abstractmethod
    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        """
        Prior plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            figure y-limits, by default (-1, 15.1)
        """
        # prior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        if ylim is not None:
            ax1.set_ylim(*ylim)
        ax1.set_xlabel("R/kpc")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        self.plot_predictive(xmodel="R", ymodel=f"{self._folded_qtys[0]}_prior", xobs="R", yobs="log10_proj_density_mean", yobs_err="log10_proj_density_std", ax=ax1)

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        fig1 = ax1[0,0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"), fig=fig1)



class GrahamModelSimple(_GrahamModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.figname_base = f"{self.figname_base}-simple"


    def extract_data(self, pars, d=None):
        """
        See docs for `_GrahamModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        super().extract_data(pars, d)
        self.transform_obs("log10_proj_density", "log10_proj_density_mean", lambda x: np.nanmean(x, axis=0))
        self.transform_obs("log10_proj_density", "log10_proj_density_std", lambda x: np.nanstd(x, axis=0))
        self.collapse_observations(["R", "log10_R", "log10_proj_density_mean", "log10_proj_density_std"])
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/{self.merger_id}-simple")


    def read_data_from_txt(self, fname, mergerid):
        """
        Read data from a txt file with columns `radius` and `surface density`.

        Parameters
        ----------
        fname : str, path-like
            data file
        mergerid : str
            merger id to be used in figure names etc.
        """
        d = self._get_data_dir(fname)
        if self._loaded_from_file:
            fname = d[0]
        _logger.info(f"Loading file: {fname}")
        data = np.loadtxt(fname)
        obs = {"R":[], "proj_density":[]}
        obs["R"] = [data[:,0]]
        obs["proj_density"] = [data[:,1]]
        self._merger_id = mergerid
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        # some transformations we need
        self.transform_obs("R", "log10_R", lambda x: np.log10(x))
        self.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/{self.merger_id}-simple")


    def _set_stan_data_OOS(self):
        raise NotImplementedError
        return super()._set_stan_data_OOS()


    def set_stan_data(self):
        """See docs for `_GrahamModelBase.set_stan_data()"""
        super().set_stan_data()
        self.stan_data.update(dict(
            log10_surf_rho = self.obs_collapsed["log10_proj_density_mean"],
            log10_surf_rho_err = self.obs_collapsed["log10_proj_density_std"]
        ))


    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys if "err" not in k]), "group")
        self._expand_dimension(["err"], "group")
        return super().all_prior_plots(figsize, ylim)


    def all_posterior_plots(self, figsize=None, ylim=(6, 10)):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            figure y-limits, by default (6, 10)

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self.latent_qtys, labeller=self._labeller_latent)

        # posterior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"log($R$/kpc)")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_ylim(*ylim)
        # TODO scale of x axis??
        self.plot_predictive(xmodel="R", ymodel=f"{self._folded_qtys_posterior[0]}", xobs="R", yobs="log10_proj_density_mean", yobs_err="log10_proj_density_std", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax



class GrahamModelHierarchy(_GrahamModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._hyper_qtys = ["log10densb_mean", "log10densb_std", 
                            "Re_sig", 
                            "g_lam", 
                            "rb_sig", 
                            "n_mean", "n_std", 
                            "a_sig",
                            "err_mean", "err_std"]
        self._hyper_qtys_labs = [
            r"$\mu_{\log_{10}\Sigma_\mathrm{b}}$", r"$\sigma_{\log_{10}\Sigma_\mathrm{b}}$",
            r"$\sigma_{R_\mathrm{e}}$",
            r"$\lambda_\gamma$",
            r"$\sigma_{r_\mathrm{b}}$",
            r"$\mu_{n}$", r"$\sigma_{n}$",
            r"$\sigma_a$",
            r"$\mu_{\tau}$", r"$\sigma_{\tau}$"
        ]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qtys_labs)))
        self._dims_prepped = False


    def extract_data(self, pars, d=None, binary=True):
        """
        See docs for `_GrahamModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'hierarchy'
        """
        super().extract_data(pars, d, binary)
        self.collapse_observations(["R", "log10_R", "log10_proj_density"])
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/{self.merger_id}-hierarchy")


    def _set_stan_data_OOS(self, r_count=None, ngroups=None):
        rmin, rmax = super()._set_stan_data_OOS()
        if r_count is None:
            r_count = max([len(rs) for rs in self.obs["R"]]) * 10
        if ngroups is None:
            ngroups = 2 * self.stan_data["N_groups"]
        self._num_OOS = r_count * ngroups
        rs = np.geomspace(rmin, rmax, r_count)
        self.stan_data.update(dict(
            N_OOS = self.num_OOS,
            R_OOS = np.tile(rs, ngroups),
            N_groups_OOS = ngroups,
            group_idx_OOS = np.repeat(np.arange(1, ngroups+1), r_count)
        ))


    def set_stan_data(self):
        """See docs for `_GrahamModelBase.set_stan_data()"""
        super().set_stan_data()
        self.stan_data.update(dict(
            log10_surf_rho = self.obs_collapsed["log10_proj_density"]
        ))


    def _prep_dims(self):
        """
        Rename dimensions for collapsing
        """
        if not self._dims_prepped:
            _rename_dict = {}
            for k in itertools.chain(self.latent_qtys, self._latent_qtys_posterior):
                _rename_dict[f"{k}_dim_0"] = "group"
            self.rename_dimensions(_rename_dict)
            self._dims_prepped = True


    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys if "err" not in k], "group"))
        self._expand_dimension(["err"], "group")
        return super().all_prior_plots(figsize, ylim)


    def all_posterior_pred_plots(self, figsize=None, ylim=(6,10)):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            figure y-limits, by default (6, 10)

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys], "group"))
        # hyper parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(self._hyper_qtys, labeller=self._labeller_hyper)

        # posterior predictive check
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        ax1.set_xlabel(r"$R$/kpc")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        ax1.set_ylim(*ylim)
        self.plot_predictive(xmodel="R", ymodel=f"{self._folded_qtys_posterior[0]}", xobs="R", yobs="log10_proj_density", ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(self.latent_qtys, figsize=figsize, labeller=self._labeller_latent, combine_dims={"group"})
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_{self._parameter_corner_plot_counter}"), fig=fig)
        return ax


    def all_posterior_OOS_plots(self, figsize=None, ylim=(6,10)):
        """
        Posterior plots for out of sample points

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            figure y-limits, by default (6, 10)

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys_posterior], "groupOOS"))

        ax = self.parameter_corner_plot(self.latent_qtys_posterior, figsize=figsize, labeller=self._labeller_latent_posterior, combine_dims={"groupOOS"})
        fig = ax.flatten()[0].get_figure()
        # note that the plot indexing uses _parameter_corner_plot_counter, so 
        # if a predictive corner plot has been made beforehand, this number will
        # be one larger
        savefig(self._make_fig_name(self.figname_base, f"corner_OOS_{self._parameter_corner_plot_counter}"), fig=fig)

        # out of sample posterior
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_xlabel(r"$R$/kpc")
        ax.set_xscale("log")
        self.posterior_OOS_plot(xmodel="R_OOS", ymodel=self._folded_qtys_posterior[0], ax=ax)
        ax.set_ylim(*ylim)
        return ax



class GrahamModelKick(_GrahamModelBase, FactorModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        _GrahamModelBase.__init__(self, model_file, prior_file, figname_base, rng)
        FactorModel_2D.__init__(self, model_file, prior_file, figname_base, rng)
        self._hyper_qtys = ["log10densb_mean", "log10densb_std",
                            "g_lam",
                            "rb_sig",
                            "n_mean", "n_std",
                            "a_sig",
                            "Re_sig",
                            "err"]
        self._hyper_qtys_labs = [r"$\mu_{\log_{10}\Sigma_\mathrm{b}}$", r"$\sigma_{\log_{10}\Sigma_\mathrm{b}}$",
                                 r"$\lambda_\gamma$",
                                 r"$\sigma_{r_\mathrm{b}}$",
                                 r"$\mu_n$", r"$\sigma_n$",
                                 r"$\sigma_a$",
                                 r"$\sigma_{R_\mathrm{e}}$",
                                 r"$\sigma$"]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qtys_labs)))
        self.rb_0 = None


    def extract_data(self, pars, d=None, binary=False):
        """
        See docs for `_GrahamModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'kick'
        """
        _GrahamModelBase.extract_data(self, pars, d, binary)
        self.rb_0 = pars["core_model_pars"]["rb_0"]["value"]
        self.figname_base = os.path.join(self.figname_base, f"{self.merger_id}/{self.merger_id}-kick")
        self.collapse_observations(["R", "log10_R", "proj_density", "log10_proj_density", "log10_proj_density_mean", "log10_proj_density_std"])


    def _set_stan_data_OOS(self, nfactors=None, ncontexts=None):
        if nfactors is None:
            nfactors = 2 * self.num_groups
            _logger.info(f"Using {nfactors} number of GQ factors")
        if ncontexts is None:
            ncontexts = 2 * self.stan_data["N_contexts"]
            _logger.info(f"Using {ncontexts} number of GQ contexts")
        rmin, rmax = super()._set_stan_data_OOS()
        r_count = max([len(rs) for rs in self.obs["R"]])
        rs = np.geomspace(rmin, rmax, r_count)
        self._num_OOS = ncontexts * r_count
        self.stan_data.update(dict(
            N_factors_OOS = nfactors,
            N_contexts_OOS = ncontexts,
            N_OOS = self.num_OOS,
            R_OOS = np.tile(rs, ncontexts),
            context_idx_OOS = np.repeat(np.arange(1, ncontexts+1), r_count),
            factor_idx_OOS = self._rng.integers(1, nfactors+1, size=ncontexts)
        ))



    def set_stan_data(self, nfactors=None, ncontexts=None):
        """
        Set the Stan data dictionary used for sampling. Setting the parameters
        to None will double the respective parameters relative to the observed
        values.

        Parameters
        ----------
        nfactors : int, optional
            number of generated quantity factors, by default None
        ncontexts : int, optional
            number of generated quantity contexts, by default None
        """
        self.stan_data = dict(
            N_tot = self.num_obs_collapsed,
            N_factors = self.num_groups,
            R = self.obs_collapsed["R"],
            log10_surf_rho = self.obs_collapsed["log10_proj_density"],
            N_contexts = sum([x.shape[0] for x in self.obs["proj_density"]])
        )
        self._set_factor_context_idxs("proj_density")
        if not self._loaded_from_file:
            self._set_stan_data_OOS(nfactors=nfactors, ncontexts=ncontexts)


    def sample_model(self, sample_kwargs={}):
        _GrahamModelBase.sample_model(self, sample_kwargs)


    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys], "group"))
        self.rename_dimensions(dict.fromkeys([f"{k}_dim_0" for k in self._hyper_qtys], "groupH"))
        fig, ax = plt.subplots(4,5,sharex="all",sharey="all")
        FactorModel_2D._plot_predictive(self, "R", "log10_surf_rho_prior", state="pred", ax=ax)
        ax[-1,-1].set_xscale("log")
        for axi in ax[-1,:]: axi.set_xlabel("R/kpc")
        for axi in ax[:,0]: axi.set_ylabel(self._folded_qtys_labs[0])

        # hyper prior corner plot
        ax1 = self.parameter_corner_plot(self._hyper_qtys, figsize=figsize, labeller=self._labeller_hyper, combine_dims={"groupH"})
        fig1 = ax1[0,0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"), fig=fig1)
        # regular prior predictive plots
        return super().all_prior_plots(figsize, None)


    def all_posterior_pred_plots(self, figsize=None, ylim=(6, 10)):
        # TODO how to handle that the folded quantity is now part of the
        # hierarchy? 
        # will potentially require rethinking how to do predictive plots
        # maybe passing a list of indices to the StanModel method?
        raise NotImplementedError
        return super().all_posterior_pred_plots(figsize, ylim)


