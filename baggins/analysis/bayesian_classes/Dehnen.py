import os
import numpy as np
import matplotlib.pyplot as plt
import pygad
from arviz.labels import MapLabeller
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.analysis.analyse_snap import basic_snapshot_centring
from baggins.mathematics import get_histogram_bin_centres, equal_count_bins
from baggins.env_config import _cmlogger, baggins_dir
from baggins.general import get_snapshot_number
from baggins.plotting import savefig

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
        self._folded_qtys = ["density"]
        self._folded_qtys_labs = [r"$\rho(r)$/(M$_\odot$/kpc$^3$))"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self._latent_qtys = ["g", "a", "err"]
        self._latent_qtys_posterior = ["g", "a", "err"]
        self._latent_qtys_labs = [r"$\gamma$", r"$a/\mathrm{kpc}$", r"$\sigma$"]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._labeller_latent_posterior = MapLabeller(
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs))
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
        d = self._get_data_dir(snapfile)
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
        self._num_OOS = r_count
        rs = np.geomspace(rmin, rmax, r_count)
        self.stan_data.update(dict(N_OOS=self.num_OOS, r_OOS=rs))

    def set_stan_data(self):
        self.stan_data = dict(
            N=self.num_obs_collapsed,
            r=self.obs_collapsed["r"],
            mass=self.obs["mass"][0][0],
            density=self.obs_collapsed["density"],
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
        fig, ax = plt.subplots(len(self.latent_qtys_posterior), 1, figsize=figsize)
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

    def all_posterior_OOS_plots(self, figsize=None):
        """
        Create out-of-sample posterior plots

        Parameters
        ----------
        ffigsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        # out of sample posterior
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$r$/kpc")
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.posterior_OOS_plot(
            xmodel="r_OOS", ymodel=self._folded_qtys_posterior[0], ax=ax
        )
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
