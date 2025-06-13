import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
from arviz.labels import MapLabeller
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.analysis.analyse_snap import basic_snapshot_centring
from baggins.mathematics import get_histogram_bin_centres, equal_count_bins
from baggins.env_config import _cmlogger, baggins_dir
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

    def extract_data(self, snapfile=None):
        obs = {"r": [], "density": [], "mass": []}
        d = self._get_data_dir(snapfile)
        if self._loaded_from_file:
            fnames = d[0]
        else:
            fnames = [snapfile]
        # is_single_file = len(fnames) == 1
        # data_ext = os.path.splitext(fnames[0])[1].lstrip(".")
        for f in fnames:
            _logger.info(f"Loading file: {f}")
        snap = pygad.Snapshot(f, physical=True)
        basic_snapshot_centring(snap)
        _logger.debug("snapshot loaded and centred")
        # TODO make this dynamic
        mask = pygad.BallMask(10)
        r_edges = equal_count_bins(snap.stars[mask]["r"], 2e5)
        obs["density"].append(
            [pygad.analysis.profile_dens(snap.stars[mask], qty="mass", r_edges=r_edges)]
        )
        obs["r"].append(get_histogram_bin_centres(r_edges))
        obs["mass"].append([np.sum(snap.stars[mask]["mass"])])
        self.obs = obs
        self.collapse_observations(["r", "density"])

    def _set_stan_data_OOS(self, r_count=None):
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
        # out of sample posterior
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$r$/kpc")
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.posterior_OOS_plot(
            xmodel="r_OOS", ymodel=self._folded_qtys_posterior[0], ax=ax
        )
        return ax
