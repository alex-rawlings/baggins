from abc import abstractmethod
import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from baggins.env_config import _cmlogger, baggins_dir
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.plotting import savefig
from baggins.utils import get_files_in_dir

__all__ = ["TerzicModelSimple"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/terzic-density/{f.rstrip('.stan')}.stan")


class _TerzicModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._folded_qtys = ["rho"]
        self._folded_qtys_labs = [r"$\rho(r)$/(M$_\odot$/kpc$^3$))"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["rb", "Re", "log10rhob", "g", "n", "a"]
        self._latent_qtys_posterior = [f"{v}_posterior" for v in self.latent_qtys]
        self._latent_qtys_labs = [
            r"$r_\mathrm{b}/\mathrm{kpc}$",
            r"$R_\mathrm{e}/\mathrm{kpc}$",
            r"$\log_{10}\left(\rho_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\gamma$",
            r"$n$",
            r"$a$",
        ]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._labeller_latent_posterior = MapLabeller(
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs))
        )
        self._merger_id = None
        self._dims_prepped = False

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
        Data extraction and manipulation required for the Terzic density model

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
        raise NotImplementedError
        obs = {"r": [], "density": []}
        d = self._get_data_dir(d)
        if self._loaded_from_file:
            fnames = d[0]
        elif os.path.isfile(d):
            fnames = [d]
        else:
            fnames = get_files_in_dir(d)
            if not fnames:
                fnames = get_files_in_dir(d, ext=".pickle")
            _logger.debug(f"Reading from dir: {d}")
        is_single_file = len(fnames) == 1
        data_ext = os.path.splitext(fnames[0])[1].lstrip(".")
        try:
            assert fnames
        except AssertionError:
            _logger.exception(
                f"Directory {d} has no files with extension {data_ext}", exc_info=True
            )
            raise
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            _data = np.loadtxt(f, skiprows=1)
            obs["r"] = _data[:, 0]
            obs["density"] = _data[:, 1]
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        if is_single_file:
            # we have loaded a single file
            # manipulate the data so it "looks" like multiple files
            _obs = obs.copy()
            obs = {"R": [], "density": []}
            for i in range(_obs["density"][0].shape[0]):
                obs["R"].append(_obs["R"][0])
                obs["density"].append(_obs["density"][0][i, :])
            _logger.warning(
                "Observations from a single file have been converted to a hierarchy format"
            )
        self.obs = obs
        # some transformations we need
        self.transform_obs("r", "log10_r", lambda x: np.log10(x))
        self.transform_obs("density", "log10_density", lambda x: np.log10(x))
        # TODO how to set the merger ID?
        self._merger_id = fnames[-1]

    @abstractmethod
    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables.
        Each derived class will need its own implementation, however all will
        require knowledge of the minimum and maximum radius to model: let's
        do that here.
        """
        rmin = np.max([r[0] for r in self.obs["r"]])
        rmax = np.min([r[-1] for r in self.obs["r"]])
        return rmin, rmax

    @abstractmethod
    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling.
        """
        self.stan_data = dict(
            N=self.num_obs_collapsed,
            N_groups=self.num_groups,
            group_idx=self.obs_collapsed["label"],
            r=self.obs_collapsed["r"],
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()

    def sample_model(self, sample_kwargs={}, diagnose=True):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS
        from previous sample.
        """
        super().sample_model(sample_kwargs=sample_kwargs, diagnose=diagnose)
        if self._loaded_from_file:
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()

    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        v = super().sample_generated_quantity(gq, force_resample, state)
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state)
            return v[..., idxs]
        else:
            return v

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
        ncol = int(np.ceil(len(self.latent_qtys) / 2))
        fig, ax = plt.subplots(2, ncol, figsize=figsize)
        try:
            self.plot_generated_quantity_dist(
                self.latent_qtys_posterior, ax=ax, xlabels=self._latent_qtys_labs
            )
        except ValueError:  # TODO check this
            _logger.warning(
                "Cannot plot latent distributions for `latent_qtys_posterior`, trying for `latent_qtys`."
            )
            self.plot_generated_quantity_dist(
                self.latent_qtys, ax=ax, xlabels=self._latent_qtys_labs
            )
        return ax

    @abstractmethod
    def all_prior_plots(self, figsize=None, ylim=None):
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
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
        if ylim is not None:
            ax1.set_ylim(*ylim)
        ax1.set_xlabel("r/kpc")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        # ax1.set_yscale("log")
        self.plot_predictive(
            xmodel="r",
            ymodel=f"log10_{self._folded_qtys[0]}_prior",
            xobs="r",
            yobs="log10_density",
            ax=ax1,
        )

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=(len(self.latent_qtys), len(self.latent_qtys)),
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


class TerzicModelSimple(_TerzicModelBase):
    def __init__(self, figname_base, rng=None):
        super().__init__(
            model_file=get_stan_file("terzic_simple"),
            prior_file=get_stan_file("terzic_simple_prior"),
            figname_base=figname_base,
            rng=rng,
        )

    def extract_data(self, pars, d=None, binary=True):
        """
        See docs for `_TerzicModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        raise NotImplementedError
        super().extract_data(pars, d, binary)
        self.collapse_observations(["r", "log10_r", "density", "log10_density"])
        self.figname_base = os.path.join(
            self.figname_base, f"{self.merger_id}/{self.merger_id}-simple"
        )

    def read_data_from_txt(self, fname, mergerid, **kwargs):
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
        data = np.loadtxt(fname, **kwargs)
        obs = {"r": [], "density": []}
        obs["r"] = [data[:, 0]]
        obs["density"] = [data[:, 1]]
        self._merger_id = mergerid
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        # some transformations we need
        self.transform_obs("r", "log10_r", lambda x: np.log10(x))
        self.transform_obs("density", "log10_density", lambda x: np.log10(x))
        self.figname_base = os.path.join(
            self.figname_base, f"{self.merger_id}/{self.merger_id}-simple"
        )
        self.collapse_observations(["r", "log10_r", "density", "log10_density"])

    def _set_stan_data_OOS(self, r_count=None):
        rmin, rmax = super()._set_stan_data_OOS()
        if r_count is None:
            r_count = max([len(rs) for rs in self.obs["r"]]) * 10
        self._num_OOS = r_count
        rs = np.geomspace(rmin, rmax, r_count)
        self.stan_data.update(dict(N_OOS=self.num_OOS, r_OOS=rs))

    def set_stan_data(self):
        """See docs for `_GrahamModelBase.set_stan_data()"""
        super().set_stan_data()
        self.stan_data.update(dict(density=self.obs_collapsed["density"]))

    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        self.rename_dimensions(
            dict.fromkeys(
                [f"{k}_dim_0" for k in self._latent_qtys if "err" not in k], "group"
            )
        )
        # self._expand_dimension(["err"], "group")
        return super().all_prior_plots(figsize, ylim)

    def all_posterior_pred_plots(self, figsize=None):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent
        )

        # posterior predictive check
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax1.set_xlabel(r"log($R$/kpc)")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        # TODO scale of x axis??
        self.plot_predictive(
            xmodel="r",
            ymodel=f"{self._folded_qtys_posterior[0]}",
            xobs="r",
            yobs="density",
            ax=ax1,
        )

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize)

        ax = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=(len(self.latent_qtys), len(self.latent_qtys)),
            labeller=self._labeller_latent,
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
        # out of sample posterior
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$r$/kpc")
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.posterior_OOS_plot(
            xmodel="r_OOS", ymodel=self._folded_qtys_posterior[0], ax=ax
        )
        return ax
