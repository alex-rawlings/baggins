from abc import abstractmethod
import os.path
import numpy as np
import matplotlib.pyplot as plt
from arviz_base.labels import MapLabeller
import pygad
from baggins.analysis.analyse_snap import basic_snapshot_centring
from baggins.analysis.bayesian_classes.StanModel import (
    HierarchicalModel_2D,
    FactorModel_2D,
)
from baggins.mathematics import get_histogram_bin_centres, equal_count_bins
from baggins.env_config import _cmlogger, baggins_dir
from baggins.general import common_string_subgroups, get_snapshot_number
from baggins.plotting import savefig, get_all_axes_from_plot_collection

__all__ = ["_GrahamModelBase", "GrahamModelSimple", "GrahamModelHierarchy"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/core-sersic/{f.rstrip('.stan')}.stan")


class _DummyDataDict:
    def __init__(self, d):
        """
        Dummy class to get a dict object into a class that looks like a `HMQuantities` class.

        Parameters
        ----------
        d : dict
            data dictionary
        """
        try:
            assert isinstance(d, dict)
        except AssertionError:
            _logger.exception(f"Input must be a dict, not {type(d)}", exc_info=True)
            raise
        self.radial_edges = None
        self.projected_mass_density = None
        self.merger_id = None
        for k in ("radial_edges", "projected_mass_density", "merger_id"):
            try:
                setattr(self, k, d[k])
            except KeyError:
                _logger.exception(f"Key {k} missing from input dict", exc_info=True)
                raise
        self.projected_mass_density = {"t0": self.projected_mass_density}
        # XXX we're not extracting this information, set to nan
        self.escape_velocity = {"t0": [np.nan, np.nan]}
        self.merger_remnant = {"kick": np.nan}


class _GrahamModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._independent_qtys = ["R"]
        self._independent_qtys_OOS = [f"{v}_OOS" for v in self._independent_qtys]
        self.independent_qtys_labs = [r"$r/\mathrm{kpc}$"]
        self._dependent_qtys = ["density"]
        self._dependent_qtys_posterior = [
            f"{v}_posterior" for v in self._dependent_qtys
        ]
        self._dependent_qtys_prior = [f"{v}_prior" for v in self._dependent_qtys]
        self._dependent_qtys_OOS = [f"{v}_OOS" for v in self._dependent_qtys]
        self.dependent_qtys_labs = [r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))"]
        self._make_xy_labellers()
        self._latent_qtys = []
        self._latent_qtys_labs = []
        self._merger_id = None

        self._latent_qtys_posterior = [f"{v}_posterior" for v in self.latent_qtys]
        self._latent_qtys_labs = [
            r"$r_\mathrm{b}/\mathrm{kpc}$",
            r"$R_\mathrm{e}/\mathrm{kpc}$",
            r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
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

    @property
    def dependent_qtys(self):
        return self._dependent_qtys

    @property
    def dependent_qtys_posterior(self):
        return self._dependent_qtys_posterior

    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    @property
    def latent_qtys_labs(self):
        return self._latent_qtys_labs

    @property
    def merger_id(self):
        return self._merger_id

    @merger_id.setter
    def merger_id(self, v):
        self._merger_id = v

    # ----------------------------------------------------------------------
    # Abstract methods
    # ----------------------------------------------------------------------

    @abstractmethod
    def extract_data(self, snapfile, extent=10, bin_count=2e5, proj=0):
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
        raise NotImplementedError

    def _prep_OOS_radii(self, r_count=None, rmin=None, rmax=None):
        """
        Set the out-of-sample Stan data variables.
        Each derived class will need its own implementation, however all will
        require knowledge of the minimum and maximum radius to model: let's
        do that here.

        Parameters
        ----------
        r_count : int, optional
            number of OOS points, by default None
        rmin : float, optional
            minimum radius, by default None
        rmax : float, optional
            maximum radius, by default None

        Returns
        -------
        rmin : float
            minimum OOS radius
        rmax : float
            maximum OOS radius
        rcount : number of OOS bins (if a new sample, else will be updated in child methods)
        """
        if r_count is None:
            r_count = max(max([len(rs) for rs in self.obs["r"]]) * 10, 500)
        _rmin = np.max([r[0] for r in self.obs["r"]])
        _rmax = np.min([r[-1] for r in self.obs["r"]])
        if rmin is None:
            rmin = _rmin
        if rmax is None:
            rmax = _rmax
        return rmin, rmax, r_count

    def _set_stan_data_OOS(self, N):
        """
        Set the out-of-sample Stan data variables.
        Each derived class will need its own implementation, however all will
        require knowledge of the minimum and maximum radius to model: let's
        do that here.
        """
        return super()._set_stan_data_OOS(N)

    @abstractmethod
    def set_stan_data(self, **kwargs):
        """
        Set the Stan data dictionary used for sampling.
        """
        if self.stan_data is None:
            self.stan_data = {}
        self.stan_data.update(
            {
                "N_obs": self.num_obs_collapsed,
                self._independent_qtys[0]: self.obs_collapsed[
                    self._independent_qtys[0]
                ],
                self._dependent_qtys[0]: self.obs_collapsed[self._dependent_qtys[0]],
            }
        )
        self._set_stan_data_OOS(**kwargs)

    @abstractmethod
    def diagnose_sample(self, var_names):
        return super().diagnose_sample(var_names)

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------

    def sample_model(self, sample_kwargs=None, diagnose=True):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS
        from previous sample.
        """
        super().sample_model(sample_kwargs=sample_kwargs, diagnose=diagnose)

    # ----------------------------------------------------------------------
    # Plotting methods
    # ----------------------------------------------------------------------

    def plot_latent_distributions(self, save=True):
        """
        Plot distributions of the latent parameters of the model

        Parameters
        ----------
        save : bool, optional
            save the figure, by default True

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        pc = self.plot_generated_quantity_dist(
            self.latent_qtys,
            labeller=self._labeller_latent,
            sample_dims=["chain", "draw", "N_groups"],
        )
        ax = get_all_axes_from_plot_collection(pc)
        fig = ax[0].get_figure()
        fig.suptitle("Latent parameters (in-sample)")
        if save:
            savefig(next(self.gen_gq_plot_name))
        return ax

    def plot_posterior_predictive(self, save=True, **kwargs):
        """
        Plot posterior predictive regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        ax : matplotlib.Axes.axes
            plotting axes
        """
        pc = super().plot_posterior_predictive(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_postpred_plot_name))
        return ax

    def plot_prior_predictive(self, save=True, **kwargs):
        pc = super().plot_prior_predictive(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_priorpred_plot_name))
        return ax

    def plot_posterior_OOS(self, save=True, **kwargs):
        """
        Plot posterior out-of-sample regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        ax : matplotlib.Axes.axes
            plotting axes
        """
        pc = super().plot_posterior_OOS(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_postOOS_plot_name))
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
        self.plot_prior_predictive()

        # prior latent quantities
        self.plot_latent_distributions()
        pc = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=(len(self.latent_qtys), len(self.latent_qtys)),
            labeller=self._labeller_latent,
            combine_dims={"group"},
        )
        fig = pc.get_viz("figure")
        savefig(next(self.gen_corner_plot_name), fig=fig)

    @abstractmethod
    def all_posterior_pred_plots(self, figsize=None):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        # posterior predictive check
        self.plot_posterior_predictive()

        # latent parameter distributions
        self.plot_latent_distributions()
        pc = self.plot_generated_quantity_dist(
            self.latent_qtys_posterior,
            labeller=self._labeller_latent_posterior,
            sample_dims=["chain", "draw", "N_groups"],
        )
        fig = pc.get_viz("figure")
        fig.suptitle("Latent parameters (out-sample)")
        savefig(next(self.gen_gq_plot_name))

        # transformed latent parameter distributions
        pc = self.parameter_corner_plot(
            self.latent_qtys_posterior,
            figsize=(len(self.latent_qtys_posterior), len(self.latent_qtys_posterior)),
            labeller=self._labeller_latent_posterior,
            combine_dims=["N_groups"],
        )
        fig = pc.get_viz("figure")
        fig.suptitle("Latent parameters (out-sample)")
        savefig(next(self.gen_corner_plot_name), fig=fig)


class GrahamModelSimple(_GrahamModelBase):
    def __init__(self, figname_base, rng=None):
        """
        Model core-Sersic projected density profile, assuming non-hierarchical
        structure of data.

        Parameters
        ----------
        figname_base : str
            path-like base name that all plots will share
        rng :  np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        super().__init__(
            model_file=get_stan_file("graham_simple"),
            prior_file=get_stan_file("graham_simple_prior"),
            figname_base=figname_base,
            rng=rng,
        )
        self._latent_qtys = [
            "log10densb",
            "log10rb",
            "log10g",
            "log10n",
            "log10a",
            "log10re",
            "err",
        ]
        self._latent_qtys_posterior = ["log10densb", "rb", "g", "n", "a", "Re", "err"]
        self._latent_qtys_labs = [
            r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
            r"$\log_{10}(r_\mathrm{b}/\mathrm{kpc})$",
            r"$\log_{10}(\gamma)$",
            r"$\log_{10}(n)$",
            r"$\log_{10}(a)$",
            r"$\log_{10}(R_\mathrm{e}/\mathrm{kpc})$",
            r"$\tau$",
        ]
        self._latent_qtys_posterior_labs = [
            r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
            r"$r_\mathrm{b}/\mathrm{kpc}$",
            r"$\gamma$",
            r"$n$",
            r"$a$",
            r"$R_\mathrm{e}/\mathrm{kpc}$",
            r"$\tau$",
        ]
        self._make_latent_labellers()

    def _make_default_merger_id(self, snapfile):
        """
        Make the default merger ID for a system if not set manually.

        Parameters
        ----------
        snapfile : str
            snapshot file name
        """
        snapnum = get_snapshot_number(snapfile)
        # use the directory name of the simulation, assumes file path is of the form:
        # /path/to/simulation/dname/output/snap_XXX.hdf5
        dname = os.path.abspath(snapfile).split("/")[-3]
        self.merger_id = f"{dname}_{snapnum}"
        _logger.warning(f"Merger ID set to the default value of {self.merger_id}")

    def extract_data(self, snapfile, extent=10, bin_count=200000, proj=0):
        obs = {"R": [], "density": []}
        d = self._get_data_files(snapfile)
        if self._loaded_from_file:
            fname = d[0]
            extent = self._input_data_and_pars["data_opts"]["extent"]
            bin_count = self._input_data_and_pars["data_opts"]["bin_count"]
            proj = self._input_data_and_pars["data_opts"]["proj"]
        else:
            fname = snapfile
            self._input_data_and_pars["data_opts"] = dict(
                extent=extent, bin_count=bin_count, proj=proj
            )
        mask = pygad.BallMask(extent)
        _logger.info(f"Loading file: {fname}")
        if self.merger_id is None:
            self._make_default_merger_id(fname)
        snap = pygad.Snapshot(fname, physical=True)
        basic_snapshot_centring(snap)
        _logger.debug("snapshot loaded and centred")
        _xy = list({0, 1, 2} - {proj})
        R = pygad.utils.geo.dist(snap.stars[mask]["pos"][_xy])
        r_edges = equal_count_bins(R, bin_count)
        obs["density"].append(
            [
                pygad.analysis.profile_dens(
                    snap.stars[mask], qty="mass", r_edges=r_edges, proj=proj
                )
            ]
        )
        obs["R"].append(get_histogram_bin_centres(r_edges, R))
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        self.collapse_observations(["R", "density"])

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
        obs = {"R": [], "proj_density": []}
        obs["R"] = [data[:, 0]]
        obs["proj_density"] = [data[:, 1]]
        self._merger_id = mergerid
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        # some transformations we need
        self.transform_obs("R", "log10_R", lambda x: np.log10(x))
        self.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))
        self.figname_base = os.path.join(
            self.figname_base, f"{self.merger_id}/{self.merger_id}-simple"
        )

    def _set_stan_data_OOS(self, r_count=None, rmin=None, rmax=None):
        """
        Set OOS Stan data.

        Parameters
        ----------
        r_count : int, optional
            Number of radii for OOS plots, by default None
        rmin : float, optional
            minimum radius, by default None
        rmax : float, optional
            maximum radius, by default None
        """
        rmin, rmax, r_count = super()._prep_OOS_radii(
            r_count=r_count, rmin=rmin, rmax=rmax
        )
        OOS_data = super()._set_stan_data_OOS(r_count)
        self._add_OOS_pars_for_saving(OOS_data)
        Rs = np.geomspace(rmin, rmax, self.num_OOS)
        OOS_data.update({self._independent_qtys_OOS[0]: Rs})
        self.stan_data.update(OOS_data)

    def set_stan_data(self):
        """See docs for _GrahamModelBase.set_stan_data()"""
        super().set_stan_data()

    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
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
        super().all_posterior_pred_plots(figsize=figsize)


class GrahamModelHierarchy(_GrahamModelBase):
    def __init__(self, figname_base, rng=None) -> None:
        """
        Model core-Sersic projected density profile, assuming hierarchical
        structure of data (i.e. from different projections).

        Parameters
        ----------
        figname_base : str
            path-like base name that all plots will share
        rng :  np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        super().__init__(
            model_file=get_stan_file("graham_hierarchy"),
            prior_file=get_stan_file("graham_hierarchy_prior"),
            figname_base=figname_base,
            rng=rng,
        )
        self._hyper_qtys = [
            "log10densb_mean",
            "log10densb_std",
            "log10densb_std",
            "log10rb_mean",
            "log10rb_std",
            "log10g_mean",
            "log10g_std",
            "log10n_mean",
            "log10n_std",
            "log10a_mean",
            "log10a_std",
            "log10Re_mean",
            "log10Re_std",
            "err",
        ]
        self._hyper_qtys_labs = [
            r"$\mu_{\log_{10}\Sigma_\mathrm{b}}$",
            r"$\sigma_{\log_{10}\Sigma_\mathrm{b}}$",
            r"$\mu_{\log_{10}r_\mathrm{b}}$",
            r"$\sigma_{\log_{10}r_\mathrm{b}}$",
            r"$\mu_{\log_{10}\gamma}$",
            r"$\sigma_{\log_{10}\gamma}$",
            r"$\mu_{\log_{10}n}$",
            r"$\sigma_{\log_{10}n}$",
            r"$\mu_{\log_{10}\alpha}$",
            r"$\sigma_{\log_{10}\alpha}$",
            r"$\mu_{\log_{10}R_\mathrm{e}}$",
            r"$\sigma_{\log_{10}R_\mathrm{e}}$",
            r"$\tau$",
        ]
        self._latent_qtys = [
            "log10densb",
            "log10rb",
            "log10g",
            "log10n",
            "log10a",
            "log10re",
        ]
        self._latent_qtys_posterior = ["log10densb", "rb", "g", "n", "a", "Re"]
        self._latent_qtys_labs = [
            r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
            r"$\log_{10}(r_\mathrm{b}/\mathrm{kpc})$",
            r"$\log_{10}(\gamma)$",
            r"$\log_{10}(n)$",
            r"$\log_{10}(a)$",
            r"$\log_{10}(R_\mathrm{e}/\mathrm{kpc})$",
        ]
        self._latent_qtys_posterior_labs = [
            r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
            r"$r_\mathrm{b}/\mathrm{kpc}$",
            r"$\gamma$",
            r"$n$",
            r"$a$",
            r"$R_\mathrm{e}/\mathrm{kpc}$",
        ]
        self._make_latent_labellers()
        self._labeller_hyper = MapLabeller(
            dict(zip(self._hyper_qtys, self._hyper_qtys_labs))
        )

    def extract_data(self, snapfiles, extent=10, bin_count=200000, proj=0):
        obs = {"R": [], "density": []}
        d = self._get_data_files(snapfiles)
        if self._loaded_from_file:
            extent = self._input_data_and_pars["data_opts"]["extent"]
            bin_count = self._input_data_and_pars["data_opts"]["bin_count"]
            proj = self._input_data_and_pars["data_opts"]["proj"]
        else:
            self._input_data_and_pars["data_opts"] = dict(
                extent=extent, bin_count=bin_count, proj=proj
            )
        mask = pygad.BallMask(extent)
        self._merger_id = os.path.splitext(
            common_string_subgroups([os.path.basename(f) for f in d])
        )[0]
        for fname in d:
            _logger.info(f"Loading file: {fname}")
            snap = pygad.Snapshot(fname, physical=True)
            basic_snapshot_centring(snap)
            _logger.debug("snapshot loaded and centred")
            _xy = list({0, 1, 2} - {proj})
            R = pygad.utils.geo.dist(snap.stars[mask]["pos"][_xy])
            r_edges = equal_count_bins(R, bin_count)
            obs["density"].append(
                [
                    pygad.analysis.profile_dens(
                        snap.stars[mask], qty="mass", r_edges=r_edges
                    )
                ]
            )
            obs["R"].append(get_histogram_bin_centres(r_edges, R))
            if not self._loaded_from_file:
                self._add_input_data_file(fname)
        self.obs = obs
        self.collapse_observations(["R", "density"])

    def _set_stan_data_OOS(self, r_count=None, rmin=None, rmax=None, ngroups=None):
        """
        Set OOS Stan data.

        Parameters
        ----------
        r_count : int, optional
            Number of radii for OOS plots, by default None
        rmin : float, optional
            minimum radius, by default None
        rmax : float, optional
            maximum radius, by default None
        ngroups : int, optional
            number of level groups (i.e. profiles), by default None
        """
        rmin, rmax, r_count = super()._prep_OOS_radii(
            r_count=r_count, rmin=rmin, rmax=rmax
        )
        OOS_data = super()._set_stan_data_OOS(r_count)
        OOS_data.setdefault(
            "N_group_OOS", 2 * self.stan_data["N_group"] if ngroups is None else ngroups
        )
        self._num_groups_OOS = OOS_data["N_group_OOS"]
        self._add_OOS_pars_for_saving(OOS_data)
        Rs = np.geomspace(rmin, rmax, self.num_OOS)
        OOS_data.update(
            {self._independent_qtys_OOS[0]: np.tile(Rs, self._num_groups_OOS)}
        )
        # update num_OOS to account for different groups
        self._num_OOS = self.num_OOS * self._num_groups_OOS
        OOS_data["N_OOS"] = self.num_OOS
        OOS_data["group_id_OOS"] = np.repeat(
            np.arange(1, self._num_groups_OOS + 1), len(Rs)
        )
        self.stan_data.update(OOS_data)

    def set_stan_data(self, **kwargs):
        """
        Set Stan data for the model.
        """
        self.stan_data.update(
            {"N_group": self.num_groups, "group_id": self.obs_collapsed["label"]}
        )
        super().set_stan_data(**kwargs)

    def diagnose_sample(self):
        return super().diagnose_sample(self._hyper_qtys)

    def all_prior_plots(self, figsize=None, ylim=None):
        """
        Make prior predictive plots for model.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            y-limits for prior predictive plot, by default None
        """
        ax = self.parameter_corner_plot(
            self._hyper_qtys, labeller=self._labeller_hyper, figsize=(8, 8)
        )
        fig = ax[0, 0].get_figure()
        savefig(next(self.corner_plot_gen), fig=fig)
        super().all_prior_plots(figsize, ylim)

    def all_posterior_pred_plots(self, figsize=None):
        """
        Make posterior predictive plots for model.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(
            self._hyper_qtys, labeller=self._labeller_hyper, figsize=(8, 8)
        )
        super().all_posterior_pred_plots(figsize)

    def all_posterior_OOS_plots(self, save=True, **kwargs):
        """
        Make posterior OOS plots for model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
         matplotlib.axes.Axes, optional
            plotting axes
        """
        return super().plot_posterior_OOS(save, **kwargs)


class GrahamModelKick(_GrahamModelBase, FactorModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        raise NotImplementedError
        _GrahamModelBase.__init__(self, model_file, prior_file, figname_base, rng)
        FactorModel_2D.__init__(self, model_file, prior_file, figname_base, rng)
        self._hyper_qtys = [
            "log10densb_mean",
            "log10densb_std",
            "g_lam",
            "rb_sig",
            "n_mean",
            "n_std",
            "a_sig",
            "Re_sig",
            "err",
        ]
        self._hyper_qtys_labs = [
            r"$\mu_{\log_{10}\Sigma_\mathrm{b}}$",
            r"$\sigma_{\log_{10}\Sigma_\mathrm{b}}$",
            r"$\lambda_\gamma$",
            r"$\sigma_{r_\mathrm{b}}$",
            r"$\mu_n$",
            r"$\sigma_n$",
            r"$\sigma_a$",
            r"$\sigma_{R_\mathrm{e}}$",
            r"$\sigma$",
        ]
        self._labeller_hyper = MapLabeller(
            dict(zip(self._hyper_qtys, self._hyper_qtys_labs))
        )

    def extract_data(self, pars, d=None, binary=False):
        """
        See docs for `_GrahamModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'kick'
        """
        _GrahamModelBase.extract_data(self, pars, d, binary)
        self.figname_base = os.path.join(
            self.figname_base, f"{self.merger_id}/{self.merger_id}-kick"
        )
        self.collapse_observations(
            [
                "R",
                "log10_R",
                "proj_density",
                "log10_proj_density",
                "log10_proj_density_mean",
                "log10_proj_density_std",
            ]
        )

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
        self.stan_data.update(
            dict(
                N_factors_OOS=nfactors,
                N_contexts_OOS=ncontexts,
                N_OOS=self.num_OOS,
                R_OOS=np.tile(rs, ncontexts),
                context_idx_OOS=np.repeat(np.arange(1, ncontexts + 1), r_count),
                factor_idx_OOS=self._rng.integers(1, nfactors + 1, size=ncontexts),
            )
        )

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
            N_tot=self.num_obs_collapsed,
            N_factors=self.num_groups,
            R=self.obs_collapsed["R"],
            log10_surf_rho=self.obs_collapsed["log10_proj_density"],
            N_contexts=sum([x.shape[0] for x in self.obs["proj_density"]]),
        )
        self._set_factor_context_idxs("proj_density")
        if not self._loaded_from_file:
            self._set_stan_data_OOS(nfactors=nfactors, ncontexts=ncontexts)

    def sample_model(self, sample_kwargs={}):
        _GrahamModelBase.sample_model(self, sample_kwargs)

    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        self.rename_dimensions(
            dict.fromkeys([f"{k}_dim_0" for k in self._latent_qtys], "group")
        )
        self.rename_dimensions(
            dict.fromkeys([f"{k}_dim_0" for k in self._hyper_qtys], "groupH")
        )
        fig, ax = plt.subplots(4, 5, sharex="all", sharey="all")
        FactorModel_2D._plot_predictive(
            self, "R", "log10_surf_rho_prior", state="pred", ax=ax
        )
        ax[-1, -1].set_xscale("log")
        for axi in ax[-1, :]:
            axi.set_xlabel("R/kpc")
        for axi in ax[:, 0]:
            axi.set_ylabel(self._folded_qtys_labs[0])

        # hyper prior corner plot
        ax1 = self.parameter_corner_plot(
            self._hyper_qtys,
            figsize=figsize,
            labeller=self._labeller_hyper,
            combine_dims={"groupH"},
        )
        fig1 = ax1[0, 0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"
            ),
            fig=fig1,
        )
        # regular prior predictive plots
        return super().all_prior_plots(figsize, None)

    def all_posterior_pred_plots(self, figsize=None, ylim=(6, 10)):
        # TODO how to handle that the folded quantity is now part of the
        # hierarchy?
        # will potentially require rethinking how to do predictive plots
        # maybe passing a list of indices to the StanModel method?
        raise NotImplementedError
        return super().all_posterior_pred_plots(figsize, ylim)
