import os
from copy import copy
import numpy as np
import pygad
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.analysis.analyse_snap import basic_snapshot_centring
from baggins.mathematics import get_histogram_bin_centres, radial_bins_by_count
from baggins.env_config import _cmlogger, baggins_dir
from baggins.general import get_snapshot_number
from baggins.plotting import savefig, get_all_axes_from_plot_collection

__all__ = ["DehnenModel"]


_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/dehnen/{f.replace('.stan', '')}.stan")


class DehnenModel(HierarchicalModel_2D):
    def __init__(self, figname_base, rng=None):
        super().__init__(
            model_file=get_stan_file("dehnen"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._independent_qtys = ["r"]
        self._fixed_parameter = ["mass"]
        self._independent_qtys_OOS = ["r_OOS"]
        self.independent_qtys_labs = [r"$r/\mathrm{kpc}$"]
        self._dependent_qtys = ["density"]
        self._dependent_qtys_posterior = [
            f"{v}_posterior" for v in self._dependent_qtys
        ]
        self._dependent_qtys_prior = [f"{v}_prior" for v in self._dependent_qtys]
        self._dependent_qtys_OOS = [f"{v}_OOS" for v in self._dependent_qtys]
        self.dependent_qtys_labs = [r"$\rho/(\mathrm{M}_\odot\,\mathrm{kpc}^{-3})$"]
        self._make_xy_labellers()
        self._latent_qtys = ["log10g", "log10a", "err"]
        self._latent_qtys_posterior = ["g", "a", "err"]
        self._latent_qtys_OOS = copy(self.latent_qtys_posterior)
        self._latent_qtys_labs = [
            r"$\log_{10}(\gamma)$",
            r"$\log_{10}(a/\mathrm{kpc})$",
            r"$\tau$",
        ]
        self._latent_qtys_posterior_labs = [r"$\gamma$", r"$a/\mathrm{kpc}$", r"$\tau$"]
        self._make_latent_labellers()
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
    def dependent_qtys_OOS(self):
        return self._dependent_qtys_OOS

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
    # Data methods
    # ----------------------------------------------------------------------

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

    def extract_data(
        self,
        snapfile=None,
        extent=10,
        n_start=100,
        n_end=10000,
        n_bins=20,
        family="stars",
    ):
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

        Returns
        -------
        snap : pygad.Snapshot
            snapshot used for fitting
        """
        obs = {"r": [], "density": [], "mass": []}
        d = self._get_data_files(snapfile)
        if self._loaded_from_file:
            fname = d[0]
            extent = self._input_data_and_pars["data_opts"]["extent"]
            family = self._input_data_and_pars["data_opts"]["family"]
            n_start = self._input_data_and_pars["data_opts"]["n_start"]
            n_end = self._input_data_and_pars["data_opts"]["n_end"]
            n_bins = self._input_data_and_pars["data_opts"]["n_bins"]
        else:
            fname = snapfile
            self._input_data_and_pars["data_opts"] = dict(
                extent=extent,
                n_start=n_start,
                n_end=n_end,
                n_bins=n_bins,
                family=family,
            )
        mask = pygad.BallMask(extent)
        _logger.info(f"Loading file: {fname}")
        if self.merger_id is None:
            self._make_default_merger_id(fname)
        snap = pygad.Snapshot(fname, physical=True)
        basic_snapshot_centring(snap)
        _logger.debug("snapshot loaded and centred")
        subsnap = getattr(snap, family)
        r_edges = radial_bins_by_count(subsnap[mask]["r"], n_start, n_end, n_bins)[0]
        if len(r_edges) < 3:
            _logger.warning("There are less than 2 data bins!")
        obs["density"].append(
            [pygad.analysis.profile_dens(subsnap[mask], qty="mass", r_edges=r_edges)]
        )
        obs["r"].append(get_histogram_bin_centres(r_edges, subsnap[mask]["r"]))
        obs["mass"].append([np.sum(subsnap["mass"])])
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        self.collapse_observations(["r", "density"])
        self.figname_base = os.path.join(
            super().figname_base, f"{self.merger_id}/{self.merger_id}"
        )
        return snap

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

    def _set_stan_data_OOS(self, r_count=None, rmin=None, rmax=None):
        rmin, rmax, r_count = self._prep_OOS_radii(
            r_count=r_count, rmin=rmin, rmax=rmax
        )
        OOS_data = super()._set_stan_data_OOS(r_count)
        rs = np.geomspace(rmin, rmax, self.num_OOS)
        OOS_data.update({self._independent_qtys_OOS[0]: rs})
        self.stan_data.update(OOS_data)

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
                self._fixed_parameter[0]: self.obs[self._fixed_parameter[0]][0][0],
                self._dependent_qtys[0]: self.obs_collapsed[self._dependent_qtys[0]],
            }
        )
        self._set_stan_data_OOS(**kwargs)

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------

    def sample_model(self, sample_kwargs={}, diagnose=True):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS
        from previous sample.
        """
        super().sample_model(sample_kwargs=sample_kwargs, diagnose=diagnose)

    def diagnose_sample(self):
        return super().diagnose_sample(self.latent_qtys)

    # ----------------------------------------------------------------------
    # Transformed quantities
    # ----------------------------------------------------------------------

    def calculate_mass_profile(self, use_OOS=False, as_xarray=False, OOS_data=None):
        """
        Calculate the mass profile from the density profile.

        Parameters
        ----------
        use_OOS : bool, optional
            use OOS quantities, by default False
        as_xarray : bool, optional
            return as xr.DataSet, by default False
        OOS_data : dict, optional
            custom OOS data, by default None

        Returns
        -------
        : np.array | xr.DataSet
            mass profile
        """
        f = lambda p, r: 4 * np.pi * p * r**2

        if use_OOS:
            if OOS_data is None:
                return f(
                    self.sample_generated_quantity(
                        self.dependent_qtys_OOS[0], as_xarray=as_xarray
                    ),
                    self.access_independent_qty(
                        self._independent_qtys_OOS[0], as_xarray=as_xarray
                    ),
                )
            else:
                for k, v in OOS_data.items():
                    if len(v) > self.num_OOS:
                        OOS_data[k] = self._rng.choice(
                            v, replace=False, size=self.num_OOS
                        )
                return f(
                    self.sample_generated_quantity_custom_OOS(
                        self.dependent_qtys_OOS[0], data=OOS_data, as_xarray=as_xarray
                    ),
                    OOS_data[self._independent_qtys_OOS[0]],
                )
        else:
            return f(
                self.sample_generated_quantity(
                    self.dependent_qtys_posterior[0], as_xarray=as_xarray
                ),
                self.access_independent_qty(
                    self._independent_qtys[0], as_xarray=as_xarray
                ),
            )

    def calculate_half_mass_radius(self, projected=False, as_xarray=False):
        """
        Calculate the mass profile from the density profile.

        Parameters
        ----------
        projected : bool, optional
            projected half mass radius, by default False
        as_xarray : bool, optional
            return as xr.DataSet, by default False

        Returns
        -------
        : np.array
            half mass radius
        """
        multiplier = 0.75 if projected else 1.0
        f = lambda a, g: multiplier * a / (2 ** (1 / (3 - g)) - 1)

        return f(
            self.sample_generated_quantity("a", as_xarray=as_xarray),
            self.sample_generated_quantity("g", as_xarray=as_xarray),
        )

    # ----------------------------------------------------------------------
    # Plotting methods
    # ----------------------------------------------------------------------

    def plot_latent_distributions(self, save=True):
        """
        Plot distributions of the latent parameters of the model.

        Parameters
        ----------
        sample_dims : list
            sampling dimensions
        save : bool, optional
            save the figure, by default True

        Returns
        -------
        pc : arviz.PlotCollection
            plotting collection
        """
        pc = self.plot_generated_quantity_dist(
            self.latent_qtys,
            labeller=self._labeller_latent,
        )
        ax = get_all_axes_from_plot_collection(pc)
        fig = ax[0].get_figure()
        fig.suptitle("Latent parameters (in-sample)")
        if save:
            savefig(next(self.gen_gq_plot_name))
        return pc

    def plot_posterior_predictive(self, save=True, **kwargs):
        """
        Plot posterior predictive regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        pc : arviz.PlotCollection
            plotting collection
        """
        pc = super().plot_posterior_predictive(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_postpred_plot_name))
        return pc

    def plot_prior_predictive(self, save=True, **kwargs):
        """
        Plot prior predictive regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        pc : arviz.PlotCollection
            plotting collection
        """
        pc = super().plot_prior_predictive(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_priorpred_plot_name))
        return pc

    def plot_posterior_OOS(self, save=True, **kwargs):
        """
        Plot posterior out-of-sample regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        pc : arviz.PlotCollection
            plotting collection
        """
        pc = super().plot_posterior_OOS(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_postOOS_plot_name))
        return pc

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
        )
        fig = pc.get_viz("figure")
        savefig(next(self.gen_corner_plot_name), fig=fig)

    def all_posterior_pred_plots(self, figsize=None):
        """
        Posterior plots generally required for predictive checks and parameter convergence

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        extra_sample_dims : list, optional
            extra sample dimensions to combine, by default None
        """
        # posterior predictive check
        self.plot_posterior_predictive()

        # latent parameter distributions
        self.plot_latent_distributions()

        sample_dims = ["chain", "draw"]
        pc = self.plot_generated_quantity_dist(
            self.latent_qtys_posterior,
            labeller=self._labeller_latent_posterior,
            sample_dims=sample_dims,
        )
        fig = pc.get_viz("figure")
        fig.suptitle("Latent parameters (out-sample)")
        savefig(next(self.gen_gq_plot_name))

        # transformed latent parameter distributions
        pc = self.parameter_corner_plot(
            self.latent_qtys_posterior,
            figsize=(len(self.latent_qtys_posterior), len(self.latent_qtys_posterior)),
            labeller=self._labeller_latent_posterior,
        )
        fig = pc.get_viz("figure")
        fig.suptitle("Latent parameters (out-sample)")
        savefig(next(self.gen_corner_plot_name), fig=fig)

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
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        except AssertionError:
            _logger.exception(f"File {fname} already exists!", exc_info=True)
            raise
        r = self.stan_data["r_OOS"]
        rho = self.sample_generated_quantity(self.dependent_qtys_OOS[0])
        pars = {}
        for p in self.latent_qtys_posterior:
            pars[p] = self.sample_generated_quantity(p)
        _logger.debug(f"r has shape {r.shape}")
        _logger.debug(f"rho has shape {rho.shape}")
        np.savez(fname, r=r, rho=rho, **pars)
        _logger.info(f"Saved OOS data to {fname}")
