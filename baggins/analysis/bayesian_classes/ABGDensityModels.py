from abc import abstractmethod
import os.path
import numpy as np
from scipy.integrate import cumulative_trapezoid
import pygad
import xarray as xr
from arviz_base.labels import MapLabeller
from baggins.env_config import _cmlogger, baggins_dir
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.analysis.analyse_snap import basic_snapshot_centring
from baggins.general import get_snapshot_number
from baggins.literature import AlphaBetaGamma_profile
from baggins.mathematics import equal_count_bins, get_histogram_bin_centres
from baggins.plotting import savefig, get_all_axes_from_plot_collection
from baggins.utils import get_files_in_dir
from baggins.general import common_string_subgroups

__all__ = ["ABGDensityModelSimple", "ABGDensityModelHierarchy"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/abg-density/{f.rstrip('.stan')}.stan")


class _ABGDensityModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        """
        Base class for the Alpha-Beta-Gamma density model.

        Parameters
        ----------
        model_file : str
            Stan model file
        prior_file : str
            Stan prior model file
        figname_base : str
            base string for figure names
        rng : np.random.Generator, optional
            random number generator, by default None
        """
        super().__init__(model_file, prior_file, figname_base, rng)
        self._independent_qtys = ["r"]
        self._independent_qtys_OOS = [f"{v}_OOS" for v in self._independent_qtys]
        self.independent_qtys_labs = [r"$r/\mathrm{kpc}$"]
        self._dependent_qtys = ["density"]
        self._dependent_qtys_posterior = [
            f"{v}_posterior" for v in self._dependent_qtys
        ]
        self._dependent_qtys_prior = [f"{v}_prior" for v in self._dependent_qtys]
        self._dependent_qtys_OOS = [f"{v}_OOS" for v in self._dependent_qtys]
        self.dependent_qtys_labs = [r"$\rho/(\mathrm{M}_\odot\,\mathrm{kpc}^{-3})$"]
        self._make_xy_labellers()
        self._dependent_qtys.append("vel_disp")
        self.dependent_qtys_labs.append(
            r"$\sigma_\mathrm{3D}/(\mathrm{km}\,\mathrm{s}^{-1})$"
        )
        self._latent_qtys = []
        self._latent_qtys_labs = []
        self._merger_id = None

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

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
    def extract_data(self):
        """
        Data extraction and manipulation required for the ABGDensity model
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
        Wrapper around StanModel.sample_model().
        """
        super().sample_model(sample_kwargs=sample_kwargs, diagnose=diagnose)

    # ----------------------------------------------------------------------
    # Plotting methods
    # ----------------------------------------------------------------------

    def plot_latent_distributions(self, save=True):
        """
        Plot distributions of the latent parameters of the model.

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
        """
        Plot prior predictive regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        ax : matplotlib.Axes.axes
            plotting axes
        """
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
        use_log = kwargs.pop("log_scale", False)
        dens_pivot = np.max(self.obs_collapsed["density"])
        _logger.debug(f"For guiding profile, pivot density is set to {dens_pivot:.3f}")
        log10dens = np.linspace(dens_pivot - offset, dens_pivot + offset, N)
        r = np.geomspace(
            np.min(self.stan_data[self._independent_qtys_OOS[0]]),
            np.max(self.stan_data[self._independent_qtys_OOS[0]]),
            500,
        )

        def profile_func(**kwargs):
            if use_log:
                return np.log10(AlphaBetaGamma_profile(**kwargs))
            else:
                return AlphaBetaGamma_profile(**kwargs)

        for p in 10**log10dens:
            label = kwargs.pop("label", None)
            ax.plot(
                r,
                profile_func(r=r, rs=rS, ps=p, a=a, b=b, g=g),
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

    def plot_velocity_dispersion_profile(self, add_obs=True):
        """
        Calculate and plot the velocity dispersion as inferred from the posterior OOS density sample.

        Parameters
        ----------
        add_obs : bool, optional
            add observed values, by default True
        """
        Gconst = 4.30091e-6  # kpc (km/s)^2 / Msun
        # wrangle with data
        r = self.stan_data[self._independent_qtys_OOS[0]]
        rho = self.sample_generated_quantity(
            self._dependent_qtys_OOS[0], as_xarray=True
        )
        rho = rho.rename_dims({f"{self._dependent_qtys_OOS[0]}_dim_0": "N_OOS"})
        rho = rho.to_dataarray()[0]
        for d in rho.dims:
            rho = rho.dropna(d)
        _logger.debug(f"rho: {rho.sizes}")

        def _reverse_cumtrapz_last(y, x):
            """
            Integral from r to rmax along the last axis. Need to negate as x is now decreasing.
            """
            return -cumulative_trapezoid(
                y[..., ::-1],
                x[::-1],
                axis=-1,
                initial=0.0,
            )[..., ::-1]

        # Enclosed mass
        radius_dim = "N_OOS"
        dMdr = 4.0 * np.pi * rho * r**2

        M = xr.apply_ufunc(
            lambda y, x: cumulative_trapezoid(y, x, axis=-1, initial=0.0),
            dMdr,
            r,
            input_core_dims=[[radius_dim], [radius_dim]],
            output_core_dims=[[radius_dim]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[rho.dtype],
        )
        assert np.all(M >= 0)

        # Jeans integrand
        integrand = rho * Gconst * M / r**2

        J = xr.apply_ufunc(
            _reverse_cumtrapz_last,
            integrand,
            r,
            input_core_dims=[[radius_dim], [radius_dim]],
            output_core_dims=[[radius_dim]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[rho.dtype],
        )
        sigma_r2 = xr.where(rho > 0, J / rho, 0.0)
        self._inference_data["predictions"]["sigma3d"] = np.sqrt(3 * sigma_r2)

        # plot
        pc = self._plot_predictive(
            x=self._independent_qtys_OOS[0],
            y="sigma3d",
            group="predictions",
            visuals={"observed_scatter": False},
        )
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_xlabel(self.independent_qtys_labs[0])
        ax.set_ylabel(self.dependent_qtys_labs[1])

        if add_obs:
            self.add_data_to_predictive_plot(
                ax=ax, xobs=self._independent_qtys[0], yobs=self._dependent_qtys[1]
            )
        savefig(next(self.gen_postOOS_plot_name))

    # ----------------------------------------------------------------------
    # Data saving
    # ----------------------------------------------------------------------

    def save_density_data_to_npz(self, dname, exist_ok=False):
        """
        Save OOS density profile to a numpy .npz file.

        Parameters
        ----------
        dname : str
            directory to save data to
        exist_ok : bool, optional
            allow overwriting
        """
        fname = os.path.join(dname, f"{self.merger_id}_density_fit.npz")
        try:
            assert not os.path.exists(fname) or exist_ok
        except AssertionError:
            _logger.exception(f"File {fname} already exists!", exc_info=True)
            raise
        r = self.stan_data[self._independent_qtys_OOS[0]]
        rho = self.sample_generated_quantity(
            self.dependent_qtys_posterior[0], state="OOS"
        )
        pars = {}
        for p in self.latent_qtys_posterior:
            pars[p] = self.sample_generated_quantity(p)
        _logger.debug(f"r has shape {r.shape}")
        _logger.debug(f"rho has shape {rho.shape}")
        np.savez(fname, r=r, rho=rho, **pars)
        _logger.info(f"Saved OOS data to {fname}")


class ABGDensityModelSimple(_ABGDensityModelBase):
    def __init__(self, figname_base, rng=None):
        """
        Construct simple model for ABG density profile.

        Parameters
        ----------
        figname_base : str
            base name for figures
        rng : numpy.random.Generator, optional
            random number generator, by default None
        """
        super().__init__(
            model_file=get_stan_file("abg_simple"),
            prior_file=get_stan_file("abg_simple_prior"),
            figname_base=figname_base,
            rng=rng,
        )
        self._latent_qtys = ["log10rS", "log10a", "b", "g", "log10rhoS", "err"]
        self._latent_qtys_posterior = ["rS", "a", "b", "g", "log10rhoS", "err"]
        self._latent_qtys_labs = [
            r"$\log_{10}(r_\mathrm{S}/\mathrm{kpc})$",
            r"$\log_{10}\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\tau$",
        ]
        self._latent_qtys_posterior_labs = [
            r"$r_\mathrm{S}/\mathrm{kpc}$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
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

    def extract_data(self, snapfile=None, extent=10, bin_count=2e5, family="stars"):
        """
        Extract data to fit from snapshot files. The snapshot is centred using the shrinking sphere method. The parameters 'extent' and 'bin_count' are saved to the data .yml files, so calling this method on a previously-fit set will use the original values.

        Parameters
        ----------
        snapfile : str, path-like, optional
            snapshot to fit, by default None
        extent : float, optional
            maximum radial extent to fit to [kpc], by default 10
        bin_count : int, float, optional
            number of stellar particles per bin, by default 2e5
        family : str, optional
            particle family to analyse, by default 'stars'
        """
        obs = {"r": [], "density": [], "mass": [], "vel_disp": []}
        d = self._get_data_files(snapfile)
        if self._loaded_from_file:
            fname = d[0]
            extent = self._input_data_and_pars["data_opts"]["extent"]
            bin_count = self._input_data_and_pars["data_opts"]["bin_count"]
            family = self._input_data_and_pars["data_opts"]["family"]
        else:
            fname = snapfile
            self._input_data_and_pars["data_opts"] = dict(
                extent=extent, bin_count=bin_count, family=family
            )
        mask = pygad.BallMask(extent)
        _logger.info(f"Loading file: {fname}")
        if self.merger_id is None:
            self._make_default_merger_id(fname)
        snap = pygad.Snapshot(fname, physical=True)
        basic_snapshot_centring(snap)
        _logger.debug("snapshot loaded and centred")
        subsnap = getattr(snap, family)
        r_edges = equal_count_bins(subsnap[mask]["r"], bin_count)
        obs["density"].append(
            [pygad.analysis.profile_dens(subsnap[mask], qty="mass", r_edges=r_edges)]
        )
        obs["vel_disp"].append(
            pygad.analysis.radially_binned_statistic(
                subsnap[mask],
                "vel",
                r_edges=r_edges,
                statistic=lambda x: np.linalg.norm(np.nanstd(x, axis=0)),
            )
        )
        obs["r"].append(get_histogram_bin_centres(r_edges, subsnap[mask]["r"]))
        obs["mass"].append([np.sum(subsnap[mask]["mass"])])
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        self.collapse_observations(["r", "density", "vel_disp"])

    def read_data_from_txt(self, fname, **kwargs):
        """
        Read data from a txt file with columns radius and surface density.

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
        rs = np.geomspace(rmin, rmax, self.num_OOS)
        OOS_data.update({self._independent_qtys_OOS[0]: rs})
        self.stan_data.update(OOS_data)

    def set_stan_data(self, **kwargs):
        """See docs for _ABGDensityModelBase.set_stan_data()"""
        super().set_stan_data(**kwargs)

    def all_prior_plots(self, figsize=None, ylim=(-1, 15.1)):
        """
        Make prior predictive plots for model.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        ylim : tuple, optional
            y-limits for prior predictive plot, by default None
        """
        return super().all_prior_plots(figsize, ylim)

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
            self.latent_qtys, labeller=self._labeller_latent, figsize=(5, 5)
        )
        super().all_posterior_pred_plots(figsize)


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
            "log10a_mean",
            "log10a_std",
            "b_mean",
            "b_std",
            "g_mean",
            "g_std",
            "err",
        ]
        self._latent_qtys = ["log10rS", "log10a", "b", "g", "log10rhoS"]
        self._latent_qtys_posterior = [
            "rS",
            "a",
            "b",
            "g",
            "log10rhoS",
        ]
        self._latent_qtys_OOS = [f"{v}_OOS" for v in self._latent_qtys_posterior]
        self._latent_qtys_labs = [
            r"$\log_{10}(r_\mathrm{S}/\mathrm{kpc})$",
            r"$\log_{10}\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\tau$",
        ]
        self._latent_qtys_posterior_labs = [
            r"$r_\mathrm{S}/\mathrm{kpc}$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\log_{10}\left(\rho_\mathrm{S}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\tau$",
        ]
        self._make_latent_labellers()
        self._hyper_qtys_labs = [
            r"$\mu_{\log_{10}\rho_\mathrm{S}}$",
            r"$\sigma_{\log_{10}\rho_\mathrm{S}}$",
            r"$\mu_{\log_{10}r_\mathrm{S}}$",
            r"$\sigma_{\log_{10}r_\mathrm{S}}$",
            r"$\mu_{\log_{10}\alpha}$",
            r"$\sigma_{\log_{10}\alpha}$",
            r"$\mu_\beta$",
            r"$\sigma_\beta$",
            r"$\mu_\gamma$",
            r"$\sigma_\gamma$",
            r"$\tau$",
        ]
        self._labeller_hyper = MapLabeller(
            dict(zip(self._hyper_qtys, self._hyper_qtys_labs))
        )

    def extract_data(self, snapfiles=None, extent=10, bin_count=2e5, family="stars"):
        """
        Extract data to fit from snapshot files. The snapshot is centred using the shrinking sphere method. The parameters 'extent' and 'bin_count' are saved to the data .yml files, so calling this method on a previously-fit set will use the original values.

        Parameters
        ----------
        snapfile : str, path-like, optional
            snapshot to fit, by default None
        extent : float, optional
            maximum radial extent to fit to [kpc], by default 10
        bin_count : int, float, optional
            number of stellar particles per bin, by default 2e5
        family : str, optional
            particle family to analyse, by default 'stars'
        """
        obs = {"r": [], "density": [], "mass": [], "vel_disp": []}
        d = self._get_data_files(snapfiles)
        if self._loaded_from_file:
            extent = self._input_data_and_pars["data_opts"]["extent"]
            bin_count = self._input_data_and_pars["data_opts"]["bin_count"]
            family = self._input_data_and_pars["data_opts"]["family"]
        else:
            self._input_data_and_pars["data_opts"] = dict(
                extent=extent, bin_count=bin_count, family=family
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
            subsnap = getattr(snap, family)
            r_edges = equal_count_bins(subsnap[mask]["r"], bin_count)
            obs["density"].append(
                [
                    pygad.analysis.profile_dens(
                        subsnap[mask], qty="mass", r_edges=r_edges
                    )
                ]
            )
            obs["vel_disp"].append(
                pygad.analysis.radially_binned_statistic(
                    subsnap[mask],
                    "vel",
                    r_edges=r_edges,
                    statistic=lambda x: np.linalg.norm(np.nanstd(x, axis=0)),
                )
            )
            obs["r"].append(get_histogram_bin_centres(r_edges, subsnap[mask]["r"]))
            obs["mass"].append([np.sum(subsnap[mask]["mass"])])
            if not self._loaded_from_file:
                self._add_input_data_file(fname)
        self.obs = obs
        self.collapse_observations(["r", "density", "vel_disp"])

    def read_data_from_txt(self, fname=None, **kwargs):
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
        if self._loaded_from_file:
            fname = self._get_data_dir(None)[0]
        if not isinstance(fname, list) and os.path.isfile(fname):
            _logger.info(f"Loading file: {fname}")
            data = np.loadtxt(fname, **kwargs)
            sample_ids = np.unique(data[:, 2])
            for _sid in sample_ids:
                mask = _sid == data[:, 2]
                obs["r"].append(data[mask, 0])
                obs["density"].append(data[mask, 1])
            self._add_input_data_file(fname)
            self._merger_id = os.path.splitext(os.path.basename(fname))[0]
        else:
            fnames = (
                fname if self._loaded_from_file else get_files_in_dir(fname, ".txt")
            )
            for _fname in fnames:
                _logger.info(f"Loading file: {_fname}")
                data = np.loadtxt(_fname, **kwargs)
                obs["r"].append(data[:, 0])
                obs["density"].append(data[:, 1])
                if not self._loaded_from_file:
                    self._add_input_data_file(_fname)
            self._merger_id = os.path.splitext(
                common_string_subgroups([os.path.basename(f) for f in fnames])
            )[0]
        _logger.debug(f"Merger ID is: {self.merger_id}")
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
        rs = np.geomspace(rmin, rmax, self.num_OOS)
        OOS_data.update(
            {self._independent_qtys_OOS[0]: np.tile(rs, self._num_groups_OOS)}
        )
        # update num_OOS to account for different groups
        self._num_OOS = self.num_OOS * self._num_groups_OOS
        OOS_data["N_OOS"] = self.num_OOS
        OOS_data["group_id_OOS"] = np.repeat(
            np.arange(1, self._num_groups_OOS + 1), len(rs)
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
