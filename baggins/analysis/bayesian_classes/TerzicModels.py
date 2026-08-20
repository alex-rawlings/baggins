import os.path
import numpy as np
import matplotlib.pyplot as plt
from arviz_base.labels import MapLabeller
import pygad
from baggins.env_config import _cmlogger, baggins_dir
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.analysis import basic_snapshot_centring
from baggins.general import get_snapshot_number
from baggins.mathematics import equal_count_bins, get_histogram_bin_centres
from baggins.plotting import savefig, plot_hdi, get_all_axes_from_plot_collection

__all__ = ["TerzicModel"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/terzic-density/{f.rstrip('.stan')}.stan")


class TerzicModel(HierarchicalModel_2D):
    def __init__(self, figname_base, rng=None) -> None:
        super().__init__(
            model_file=get_stan_file("terzic"),
            prior_file=get_stan_file("terzic_prior"),
            figname_base=figname_base,
            rng=rng,
        )
        self._folded_qtys = ["rho"]
        self._folded_qtys_labs = [r"$\rho(r)$/(M$_\odot$/kpc$^3$))"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["log10rb", "log10Re", "log10rhob", "g", "n", "a"]
        self._latent_qtys_posterior = ["rb", "Re", "log10rhob", "g", "n", "a"]
        self._latent_qtys_labs = [
            r"$\log_{10}\left(r_\mathrm{b}/\mathrm{kpc}\right)$",
            r"$\log_{10}\left(R_\mathrm{e}/\mathrm{kpc}\right)$",
            r"$\log_{10}\left(\rho_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-3})\right)$",
            r"$\gamma$",
            r"$n$",
            r"$a$",
        ]
        self._latent_qtys_posterior_labs = [
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
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_posterior_labs))
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
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    @property
    def merger_id(self):
        return self._merger_id

    @merger_id.setter
    def merger_id(self, v):
        self._merger_id = v

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

    def extract_data(self, snapfile=None, extent=10, bin_count=2e5):
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
        """
        obs = {"r": [], "density": [], "mass": []}
        d = self._get_data_files(snapfile)
        if self._loaded_from_file:
            fname = d[0][0]
            extent = self._input_data_files["kwargs"]["extent"]
            bin_count = self._input_data_files["kwargs"]["bin_count"]
        else:
            fname = snapfile
            self._input_data_files["kwargs"] = dict(extent=extent, bin_count=bin_count)
        mask = pygad.BallMask(extent)
        _logger.info(f"Loading file: {fname}")
        if self.merger_id is None:
            self._make_default_merger_id(fname)
        snap = pygad.Snapshot(fname, physical=True)
        basic_snapshot_centring(snap)
        _logger.debug("snapshot loaded and centred")
        r_edges = equal_count_bins(snap.stars[mask]["r"], bin_count)
        obs["density"].append(
            [pygad.analysis.profile_dens(snap.stars[mask], qty="mass", r_edges=r_edges)]
        )
        obs["r"].append(get_histogram_bin_centres(r_edges))
        obs["mass"].append([np.sum(snap.stars[mask]["mass"])])
        if not self._loaded_from_file:
            self._add_input_data_file(fname)
        self.obs = obs
        self.collapse_observations(["r", "density"])

    def read_data_from_txt(self, fname, **kwargs):
        """
        Read data from a txt file with columns `radius` and `surface density`.

        Parameters
        ----------
        fname : str, path-like
            data file
        """
        d = self._get_data_files(fname)
        if self._loaded_from_file:
            fname = d[0]
        _logger.info(f"Loading file: {fname}")
        data = np.loadtxt(fname, **kwargs)
        obs = {"r": [], "density": []}
        obs["r"] = [data[:, 0]]
        obs["density"] = [data[:, 1]]
        if self.merger_id is None:
            self._make_default_merger_id(fname)
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
        """
        Set the OOS Stan data.

        Parameters
        ----------
        r_count : int, optional
            number of radial points to sample, by default None
        """
        rmin = np.max([r[0] for r in self.obs["r"]])
        rmax = np.min([r[-1] for r in self.obs["r"]])
        if r_count is None:
            r_count = max([len(rs) for rs in self.obs["r"]]) * 10
        OOS_data = super()._set_stan_data_OOS(r_count)
        rs = np.geomspace(rmin, rmax, self.num_OOS)
        OOS_data.update({"r_OOS": rs})
        self.stan_data.update(OOS_data)

    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling.
        """
        self.stan_data = dict(
            N=self.num_obs_collapsed,
            r=self.obs_collapsed["r"],
            density=self.obs_collapsed["density"],
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()

    def diagnose_sample(self):
        return super().diagnose_sample(self.latent_qtys)

    def _get_GQ_indices(self, state):
        """
        The Terzic Stan model evaluates the density generated quantity at the
        concatenation of in-sample and OOS radii (length N + N_OOS). Get the
        indices of the requested subset.

        Parameters
        ----------
        state : str
            inference type, must be one of 'pred' or 'OOS'

        Returns
        -------
        slice
            index range for the requested subset
        """
        dividing_idx = self.num_obs_collapsed
        return (
            slice(None, dividing_idx)
            if state == "pred"
            else slice(dividing_idx, self.num_OOS + dividing_idx)
        )

    def sample_generated_quantity(
        self, gq, force_resample=False, state="pred", as_xarray=False
    ):
        v = super().sample_generated_quantity(
            gq, force_resample=force_resample, as_xarray=as_xarray
        )
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state)
            return v[..., idxs]
        else:
            return v

    def plot_latent_distributions(self, transformed=False, figsize=None):
        """
        Plot distributions of the latent parameters of the model

        Parameters
        ----------
        transformed: bool, optional
            plot transformed latent parameters (also called 'posterior')
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        vals = self.latent_qtys_posterior if transformed else self.latent_qtys
        try:
            pc = self.plot_generated_quantity_dist(vals)
        except ValueError:  # TODO check this
            _logger.warning(
                "Cannot plot latent distributions for `latent_qtys_posterior`, trying for `latent_qtys`."
            )
            pc = self.plot_generated_quantity_dist(self.latent_qtys)
        ax = get_all_axes_from_plot_collection(pc)
        return ax

    def _plot_predictive_1var(
        self, ymodel, state, xobs=None, yobs=None, ax=None, levels=None, collapsed=True
    ):
        """
        Plot a HDI band for a generated quantity against the observed data.

        Parameters
        ----------
        ymodel : str
            generated quantity to plot
        state : str
            'pred' for the in-sample radii, 'OOS' for the out-of-sample radii
        xobs : str, optional
            observed independent variable to overlay, by default None
        yobs : str, optional
            observed dependent variable to overlay, by default None
        ax : matplotlib.axes.Axes, optional
            axis to plot to, by default None (creates new instance)
        levels : list, optional
            HDI intervals to plot, by default None
        collapsed : bool, optional
            plot collapsed observations, by default True

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        if levels is None:
            levels = list(self._default_hdi_levels)
        x = self.stan_data["r" if state == "pred" else "r_OOS"]
        y = self.sample_generated_quantity(ymodel, state=state)
        for lev in sorted(levels):
            plot_hdi(
                x,
                y,
                hdi_prob=lev,
                ax=ax,
                plot_kwargs={"color": self.hdi_col_mapper.get_colour(lev)},
                fill_kwargs={
                    "alpha": 0.4,
                    "color": self.hdi_col_mapper.get_colour(lev),
                },
                hdi_kwargs={"skipna": True},
            )
        if xobs is not None and yobs is not None:
            obs = self.obs_collapsed if collapsed else self.obs
            ax.scatter(obs[xobs], obs[yobs], **self._plot_obs_data_kwargs)
        return ax

    def plot_prior_predictive(self, save=True, ax=None, **kwargs):
        """
        Plot the prior predictive density profile against observed data.
        """
        ax = self._plot_predictive_1var(
            ymodel=f"log10_{self.folded_qtys[0]}_prior",
            state="pred",
            xobs="r",
            yobs="log10_density",
            ax=ax,
            **kwargs,
        )
        if save:
            savefig(next(self.gen_priorpred_plot_name), fig=ax.get_figure())
        return ax

    def plot_posterior_predictive(self, save=True, ax=None, **kwargs):
        """
        Plot the posterior predictive density profile against observed data.
        """
        ax = self._plot_predictive_1var(
            ymodel=self.folded_qtys_posterior[0],
            state="pred",
            xobs="r",
            yobs="density",
            ax=ax,
            **kwargs,
        )
        if save:
            savefig(next(self.gen_postpred_plot_name), fig=ax.get_figure())
        return ax

    def plot_posterior_OOS(self, save=True, ax=None, **kwargs):
        """
        Plot the out-of-sample posterior density profile.
        """
        ax = self._plot_predictive_1var(
            ymodel=self.folded_qtys_posterior[0], state="OOS", ax=ax, **kwargs
        )
        if save:
            savefig(next(self.gen_postOOS_plot_name), fig=ax.get_figure())
        return ax

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
        self.plot_prior_predictive(ax=ax1)

        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=(len(self.latent_qtys), len(self.latent_qtys)),
            labeller=self._labeller_latent,
            combine_dims={"group"},
        )
        fig1 = ax1[0, 0].get_figure()
        savefig(next(self.gen_corner_plot_name), fig=fig1)

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
        ax1.set_xlabel(r"$r$/kpc")
        ax1.set_ylabel(self._folded_qtys_labs[0])
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        self.plot_posterior_predictive(ax=ax1)

        # latent parameter distributions
        self.plot_latent_distributions(figsize=figsize, transformed=True)

        ax = self.parameter_corner_plot(
            self.latent_qtys_posterior,
            figsize=(len(self.latent_qtys_posterior), len(self.latent_qtys_posterior)),
            labeller=self._labeller_latent_posterior,
        )
        fig = ax.flatten()[0].get_figure()
        savefig(next(self.gen_corner_plot_name), fig=fig)
        return ax

    def all_posterior_OOS_plots(self, figsize=None):
        # out of sample posterior
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$r$/kpc")
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.plot_posterior_OOS(ax=ax)
        return ax

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
        rho = self.sample_generated_quantity(self.folded_qtys_posterior[0], state="OOS")
        pars = {}
        for p in self.latent_qtys_posterior:
            pars[p] = self.sample_generated_quantity(p)
        _logger.debug(f"r has shape {r.shape}")
        _logger.debug(f"rho has shape {rho.shape}")
        np.savez(fname, r=r, rho=rho, **pars)
        _logger.info(f"Saved OOS data to {fname}")
