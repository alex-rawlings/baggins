from abc import abstractmethod
from copy import copy
import os.path
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from baggins.env_config import _cmlogger, baggins_dir
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.literature import AlphaBetaGamma_profile
from baggins.plotting import savefig
from baggins.utils import get_files_in_dir

__all__ = ["ABGDensityModelSimple", "ABGDensityModelHierarchy"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/abg-density/{f.rstrip('.stan')}.stan")


class _ABGDensityModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._folded_qtys = ["rho"]
        self._folded_qtys_labs = [r"$\rho(r)$/(M$_\odot$/kpc$^3$))"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = []
        self._latent_qtys_labs = []
        self._latent_qtys_labs = []
        self._latent_qtys_posterior_labs = []
        self._merger_id = None
        self._dims_prepped = False

    @property
    def independent_var_lab(self):
        return r"$r/\mathrm{kpc}$"

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
    def latent_qtys_labs(self):
        return self._latent_qtys_labs

    @property
    def merger_id(self):
        return self._merger_id

    def _make_latent_labellers(self):
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._labeller_latent_posterior = MapLabeller(
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_posterior_labs))
        )

    @abstractmethod
    def extract_data(self):
        """
        Data extraction and manipulation required for the ABGDensity model
        """
        raise NotImplementedError

    @abstractmethod
    def _set_stan_data_OOS(self, r_count=None, rmin=None, rmax=None):
        """
        Set the out-of-sample Stan data variables.
        Each derived class will need its own implementation, however all will
        require knowledge of the minimum and maximum radius to model: let's
        do that here.
        """
        _rmin = np.max([r[0] for r in self.obs["r"]])
        _rmax = np.min([r[-1] for r in self.obs["r"]])
        if rmin is None:
            rmin = _rmin
        if rmax is None:
            rmax = _rmax
        if r_count is None:
            r_count = max([len(rs) for rs in self.obs["r"]]) * 10
        self._num_OOS = r_count
        _logger.debug(
            f"OOS will have radial values from {rmin:.2e} - {rmax:.2e} in {r_count} bins"
        )
        rs = np.geomspace(rmin, rmax, r_count)
        return rs

    @abstractmethod
    def set_stan_data(self, **kwargs):
        """
        Set the Stan data dictionary used for sampling.
        """
        self.stan_data = dict(
            N_obs=self.num_obs_collapsed,
            N_group=self.num_groups,
            group_id=self.obs_collapsed["label"],
            r=self.obs_collapsed["r"],
            density=self.obs_collapsed["density"],
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS(**kwargs)

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
            self.rename_dimensions(
                dict.fromkeys([f"{k}_dim_0" for k in self.latent_qtys], "group")
            )
            self.rename_dimensions(
                dict.fromkeys(
                    [f"{k}_dim_0" for k in self.latent_qtys_posterior], "groupOOS"
                )
            )
            self._dims_prepped = True

    def plot_latent_distributions(self, ax=None, figsize=None, from_hyper=False):
        """
        Plot distributions of the latent parameters of the model.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            plotting axis, by default None
        figsize : tuple, optional
            figure size, by default None
        from_hyper : bool, optional
            plot latent parameters as sampled from hyperdistribution, by default False

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        if from_hyper:
            lq = self.latent_qtys_posterior
            lql = self._latent_qtys_posterior_labs
            lqstr = "latent_qtys_posterior"
        else:
            lq = self.latent_qtys
            lql = self.latent_qtys_labs
            lqstr = "latent_qtys"
        ncol = int(np.ceil(len(lq) / 2))
        if ax is None:
            fig, ax = plt.subplots(2, ncol, figsize=figsize)
        try:
            self.plot_generated_quantity_dist(
                lq,
                ax=ax,
                xlabels=lql,
            )
        except ValueError:  # TODO check this
            _logger.warning(f"Cannot plot latent distributions for '{lqstr}'.")
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
        ax1.set_xlabel(self.independent_var_lab)
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        self.plot_predictive(
            xmodel="r",
            ymodel=f"{self._folded_qtys[0]}_prior",
            xobs="r",
            yobs="density",
            ax=ax1,
        )

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax2 = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=(len(self.latent_qtys), len(self.latent_qtys)),
            labeller=self._labeller_latent,
            combine_dims={"group"},
        )
        fig2 = ax2[0, 0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"
            ),
            fig=fig2,
        )

    @abstractmethod
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
            self.latent_qtys_posterior,
            figsize=(len(self.latent_qtys_posterior), len(self.latent_qtys_posterior)),
            labeller=self._labeller_latent_posterior,
            combine_dims={"groupOOS"},
        )
        fig = ax.flatten()[0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_{self._parameter_corner_plot_counter}"
            ),
            fig=fig,
        )
        return ax

    def all_posterior_OOS_plots(self, figsize=None, ax=None):
        # out of sample posterior
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$r$/kpc")
        ax.set_ylabel(self._folded_qtys_labs[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.posterior_OOS_plot(
            xmodel="r_OOS", ymodel=self._folded_qtys_posterior[0], ax=ax
        )
        return ax

    def add_guiding_profiles(self, ax, a, b, g, rS, N=5, offset=0.5, **kwargs):
        """
        Plot some ABG profiles, varying the normalising density, to guide the eye.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            plotting axis
        a : float
            transition index
        b : float
            outer region slope
        g : float
            inner region slope
        rS : float
            scale radius
        N : int, optional
            number of lines, by default 5
        offset : float, optional
            half vertical spacing between lowest and highest profile, by default 0.5
        """
        kwargs.setdefault("lw", 1)
        kwargs.setdefault("c", "gray")
        kwargs.setdefault("zorder", 0.2)
        kwargs.setdefault("label", f"({a:.1f},{b:.1f},{g:.1f})")
        dens_pivot = np.max(self.obs["log10_density"])
        log10dens = np.linspace(dens_pivot - offset, dens_pivot + offset, N)
        r = np.geomspace(
            np.min(self.stan_data["r_OOS"]), np.max(self.stan_data["r_OOS"]), 500
        )
        for p in 10**log10dens:
            label = kwargs.pop("label", None)
            ax.plot(
                r,
                AlphaBetaGamma_profile(r, rs=rS, ps=p, a=a, b=b, g=g),
                label=label,
                **kwargs,
            )

    def add_guiding_Plummer(self, ax, rS, N=5, offset=0.5, **kwargs):
        """
        Plot Plummer profile to guide eye. See add_guiding_profiles() for details.
        """
        kwargs.setdefault("label", "Plummer")
        self.add_guiding_profiles(
            ax=ax, a=2, b=5, g=0, rS=rS, N=N, offset=offset, **kwargs
        )

    def add_guiding_NFW(self, ax, rS, N=5, offset=0.5, **kwargs):
        """
        Plot NFW profile to guide eye. See add_guiding_profiles() for details.
        """
        kwargs.setdefault("label", "NFW")
        self.add_guiding_profiles(
            ax=ax, a=1, b=3, g=1, rS=rS, N=N, offset=offset, **kwargs
        )

    def save_density_data_to_npz(self, dname, exist_ok=False):
        """
        Save OOS density profile to a numpy .npz file with keys 'x' and 'y'.

        Parameters
        ----------
        dname : directory to save data to
            file to save data to
        """
        fname = os.path.join(dname, f"{self.merger_id}_density_fit.npz")
        try:
            assert not os.path.exists(fname) or exist_ok
        except AssertionError:
            _logger.exception(f"File {fname} already exists!", exc_info=True)
            raise
        r = self.stan_data["r_OOS"]
        rho = self.sample_generated_quantity(self.folded_qtys_posterior[0], state="OOS")
        pars = {}
        for p in self.latent_qtys_posterior:
            pars[p] = self.sample_generated_quantity(p)
        _logger.debug(f"r has shape {r.shape}")
        _logger.debug(f"rho has shape {rho.shape}")
        np.savez(fname, r=r, rho=rho, **pars)
        _logger.info(f"Saved OOS data to {fname}")


class ABGDensityModelSimple(_ABGDensityModelBase):
    def __init__(self, figname_base, rng=None):
        super().__init__(
            model_file=get_stan_file("abg_simple"),
            prior_file=get_stan_file("abg_simple_prior"),
            figname_base=figname_base,
            rng=rng,
        )
        self._latent_qtys = [
            "log10rS",
            "a",
            "b",
            "g_raw",
            "log10rhoS",
            "err0",
            "err_grad",
        ]
        self._latent_qtys_posterior = [
            "rS",
            "a",
            "b",
            "g",
            "log10rhoS",
            "err0",
            "err_grad",
        ]
        self._latent_qtys_labs = [
            r"$\log_{10}(r_\mathrm{S}/\mathrm{kpc})$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma'$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\tau_0$",
            r"$\tau_\Delta$",
        ]
        self._latent_qtys_posterior_labs = [
            r"$r_\mathrm{S}/\mathrm{kpc}$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\tau_0$",
            r"$\tau_\Delta$",
        ]
        self._make_latent_labellers()

    def extract_data(self, fname, **kwargs):
        """
        See docs for `_ABGDensityModelBase.extract_data()"
        Update figname_base to include merger ID and keyword 'simple'
        """
        self.read_data_from_txt(fname=fname, **kwargs)

    def read_data_from_txt(self, fname, **kwargs):
        """
        Read data from a txt file with columns `radius` and `surface density`.

        Parameters
        ----------
        fname : str, path-like
            data file
        """
        d = self._get_data_dir(fname)
        if self._loaded_from_file:
            if os.path.isdir(d):
                fname = d[0]
            else:
                fname = d
        _logger.info(f"Loading file: {fname}")
        data = np.loadtxt(fname, **kwargs)
        obs = {"r": [], "density": []}
        obs["r"] = [data[:, 0]]
        obs["density"] = [data[:, 1]]
        self._merger_id = os.path.splitext(os.path.basename(fname))[0]
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

    def _set_stan_data_OOS(self, r_count=None, rmin=None, rmax=None):
        rs = super()._set_stan_data_OOS(r_count=r_count, rmin=rmin, rmax=rmax)
        self.stan_data.update(dict(N_OOS=self.num_OOS, r_OOS=rs))

    def set_stan_data(self, **kwargs):
        """See docs for `_ABGDensityModelBase.set_stan_data()"""
        super().set_stan_data(**kwargs)

    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        self.rename_dimensions(
            dict.fromkeys(
                [f"{k}_dim_0" for k in self._latent_qtys if "err" not in k], "group"
            )
        )
        return super().all_prior_plots(figsize, ylim)

    def all_posterior_pred_plots(self, figsize=None):
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent, figsize=(5, 5)
        )
        return super().all_posterior_pred_plots(figsize)


class ABGDensityModelHierarchy(_ABGDensityModelBase):
    def __init__(self, figname_base, rng=None):
        """
        Construct hierarchical model for ABG density profile.

        Parameters
        ----------
        figname_base : str
            base name for figures
        rng : numpy.random.Generator, optional
            random number generator, by default None
        """
        super().__init__(
            model_file=get_stan_file("abg_hierarchy"),
            prior_file=get_stan_file("abg_hierarchy_prior"),
            figname_base=figname_base,
            rng=rng,
        )
        self._hyper_qtys = [
            "log10rhoS_mean",
            "log10rhoS_std",
            "log10rS_mean",
            "log10rS_std",
            "a_mean",
            "a_std",
            "b_mean",
            "b_std",
            "g_mean",
            "g_std",
            "obs_sigma",
        ]
        self._latent_qtys = ["log10rS", "a", "b", "g", "log10rhoS"]
        self._latent_qtys_posterior = [f"{k}_posterior" for k in self._latent_qtys]
        self._latent_qtys_labs = [
            r"$\log_{10}(r_\mathrm{S}/\mathrm{kpc})$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
        ]
        self._latent_qtys_posterior_labs = copy(self._latent_qtys_labs)
        self._make_latent_labellers()
        self._hyper_qtys_labs = [
            r"$\mu_{\log_{10}\rho_\mathrm{S}}$",
            r"$\sigma_{\log_{10}\rho_\mathrm{S}}$",
            r"$\mu_{\log_{10}r_\mathrm{S}}$",
            r"$\sigma_{\log_{10}r_\mathrm{S}}$",
            r"$\mu_a$",
            r"$\sigma_a$",
            r"$\mu_b$",
            r"$\sigma_b$",
            r"$\mu_\gamma$",
            r"$\sigma_\gamma$",
            r"$\tau$",
        ]
        self._hyper_qtys_labs.extend(self._latent_qtys_labs[-2:])
        self._labeller_hyper = MapLabeller(
            dict(zip(self._hyper_qtys, self._hyper_qtys_labs))
        )

    def extract_data(self, fname=None, **kwargs):
        """
        Extract data from .txt file or a directory containing .txt files.
        Last data point is not used for the fitting.

        Parameters
        ----------
        fname : str, optional
            path to file(s), by default None
        """
        obs = {"r": [], "density": []}
        fname = fname.rstrip("/")
        fname = self._get_data_dir(fname)
        if self._loaded_from_file:
            fnames = fname[0]
        elif os.path.isfile(fname):
            _logger.info(f"Loading file: {fname}")
            data = np.loadtxt(fname, **kwargs)
            sample_ids = np.unique(data[:, 2])
            for _sid in sample_ids:
                mask = _sid == data[:, 2]
                # TODO how to exclude last point efficiently?
                obs["r"].append(data[mask, 0])
                obs["density"].append(data[mask, 1])
            self._add_input_data_file(fname)
        else:
            fnames = get_files_in_dir(fname, ".txt")
            for _fname in fnames:
                _logger.info(f"Loading file: {_fname}")
                data = np.loadtxt(_fname, **kwargs)
                # XXX here we exclude the last point
                obs["r"].append(data[:-1, 0])
                obs["density"].append(data[:-1, 1])
                if not self._loaded_from_file:
                    self._add_input_data_file(_fname)
        self._merger_id = os.path.splitext(os.path.basename(fname))[0]
        self.obs = obs
        # some transformations we need
        self.transform_obs("r", "log10_r", lambda x: np.log10(x))
        self.transform_obs("density", "log10_density", lambda x: np.log10(x))
        self.figname_base = os.path.join(
            self.figname_base, f"{self.merger_id}/{self.merger_id}-hierarchy"
        )
        self.collapse_observations(["r", "log10_r", "density", "log10_density"])

    def _set_stan_data_OOS(self, r_count=None, rmin=None, rmax=None, ngroups=None):
        """
        _summary_

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
        rs = super()._set_stan_data_OOS(r_count=r_count, rmin=rmin, rmax=rmax)
        if ngroups is None:
            ngroups = 2 * self.stan_data["N_group"]
        # update num_OOS to account for different groups
        self._num_OOS = self._num_OOS * ngroups
        self.stan_data.update(
            dict(
                N_OOS=self.num_OOS,
                r_OOS=np.tile(rs, ngroups),
                N_group_OOS=ngroups,
                group_id_OOS=np.repeat(np.arange(1, ngroups + 1), len(rs)),
            )
        )

    def set_stan_data(self, **kwargs):
        """
        Set Stan data for the model.
        """
        return super().set_stan_data(**kwargs)

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
        self._prep_dims()
        ax = self.parameter_corner_plot(
            self._hyper_qtys, labeller=self._labeller_hyper, figsize=(8, 8)
        )
        fig = ax[0, 0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_prior_{self._parameter_corner_plot_counter}"
            ),
            fig=fig,
        )
        super().all_prior_plots(figsize, ylim)

    def all_posterior_pred_plots(self, figsize=None):
        """
        Make posterior predictive plots for model.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        : matplotlib.axes.Axes
            plotting axes for latent parameter corner plot
        """
        self._prep_dims()
        self.plot_latent_distributions(figsize=figsize, from_hyper=True)
        # latent parameter plots (corners, chains, etc)
        self.parameter_diagnostic_plots(
            self._hyper_qtys, labeller=self._labeller_hyper, figsize=(8, 8)
        )
        return super().all_posterior_pred_plots(figsize)

    def all_posterior_OOS_plots(self, figsize=None, ax=None):
        """
        Make posterior OOS plots for model.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None

        Returns
        -------
         matplotlib.axes.Axes, optional
            plotting axes
        """
        self._prep_dims()
        return super().all_posterior_OOS_plots(figsize, ax)
