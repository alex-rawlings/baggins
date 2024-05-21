from abc import abstractmethod
from tqdm import tqdm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from ketjugw.units import km_per_s
from . import HierarchicalModel_2D
from ..analyse_ketju import get_bound_binary
from ...env_config import _cmlogger
from ...general.units import kpc
from ...literature import (
    zlochower_dry_spins,
    ketju_calculate_bh_merger_remnant_properties,
)
from ...mathematics import uniform_sample_sphere, convert_spherical_to_cartesian
from ...plotting import savefig
from ...utils import load_data, get_ketjubhs_in_dir


__all__ = ["CoreKickExp", "CoreKickLinear", "CoreKickSigmoid"]

_logger = _cmlogger.getChild(__name__)


class _CoreKickBase(HierarchicalModel_2D):
    def __init__(
        self,
        model_file,
        prior_file,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.escape_vel = escape_vel
        self.premerger_ketjufile = premerger_ketjufile
        self._folded_qtys = ["rb"]
        self._folded_qtys_labs = [r"$r_\mathrm{b}/r_{\mathrm{b},0}$"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]
        self.bh1 = None
        self.bh2 = None
        self._rb0 = np.nan

    @property
    def folded_qtys(self):
        return self._folded_qtys

    @property
    def folded_qtys_posterior(self):
        return self._folded_qtys_posterior

    @property
    @abstractmethod
    def latent_qtys(self):
        pass

    @property
    @abstractmethod
    def _latent_qtys_labs(self):
        pass

    @property
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    @property
    def _latent_qtys_posterior(self):
        return self.latent_qtys

    @property
    def _labeller_latent(self):
        return MapLabeller(dict(zip(self.latent_qtys, self._latent_qtys_labs)))

    @property
    def _labeller_latent_posterior(self):
        return MapLabeller(dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs)))

    @property
    def rb0(self):
        return self._rb0

    def extract_data(self, d=None, npoints=200, pars=None):
        """
        Data extraction and manipulation required by the CoreKick model.
        Due to the complexity of extracting core radius, samples from the core
        radius distribution for each kick velocity are assumed as an input
        pickle file. The structure of the file must be of the form:
        {'rb': {
                "XXXX": [core radius values],
                ...,
                "YYYY": [core radius values]
        }}
        Where XXXX and YYYY are the kick velocities as strings, convertible to
        a float (e.g. "0060").
        The `pars` parameter is unused and included only for compatability with
        the parent class.


        Parameters
        ----------
        d : path-like, optional
            file of core radius samples, by default None (paths read from
            `_input_data_files`)
        npoints : int, optional
            thin the data to this many points per kick velocity, by default 200
        """
        try:
            assert self.escape_vel is not None and self.premerger_ketjufile is not None
        except AssertionError:
            _logger.exception(
                "Attributes `escape_vel` and `premerger_ketjufile` must be set before extracting data",
                exc_info=True,
            )
            raise
        d = self._get_data_dir(d)
        data = load_data(d)
        obs = {"vkick": [], "rb": []}
        for k, v in data["rb"].items():
            if k == "__githash" or k == "__script":
                continue
            _logger.info(f"Getting data for kick {k}")
            mask = ~np.isnan(v)
            v = v[mask]
            rb0 = data["rb"]["0000"][mask]
            idxs = self._rng.choice(np.arange(len(v)), 200, replace=False)
            obs["vkick"].append([float(k) / self.escape_vel])
            obs["rb"].append([np.nanmedian(v[idxs].flatten() / rb0[idxs].flatten())])
        self._rb0 = np.nanmean(rb0)
        self.obs = obs
        if not self._loaded_from_file:
            self._add_input_data_file(d)
        self.collapse_observations(["vkick", "rb"])
        # extract BH data at the timestep before merger
        kfile = get_ketjubhs_in_dir(self.premerger_ketjufile)[0]
        bh1, bh2, *_ = get_bound_binary(kfile)
        # move to Gadget units: kpc, km/s, 1e10Msol
        bh1.x /= kpc
        bh2.x /= kpc
        bh1.v /= km_per_s
        bh2.v /= km_per_s
        bh1.m /= 1e10
        bh2.m /= 1e10
        self.bh1 = bh1[-1]
        self.bh2 = bh2[-1]

    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variables. 10000 OOS points will be
        used.
        BH spins are uniformly sampled on the sphere, with magnitude from the
        Zlochower Lousto "dry" distribution.
        """
        _OOS = {"N_OOS": 10000}
        self._num_OOS = _OOS["N_OOS"]
        t, p = uniform_sample_sphere(_OOS["N_OOS"] * 2, rng=self._rng)
        spin_mag = scipy.stats.beta.rvs(
            *zlochower_dry_spins.values(),
            random_state=self._rng,
            size=_OOS["N_OOS"] * 2,
        )
        spins = convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
        s1 = spins[: _OOS["N_OOS"], :]
        s2 = spins[_OOS["N_OOS"] :, :]
        vkick = np.full(_OOS["N_OOS"], np.nan)
        for i, (ss1, ss2) in tqdm(enumerate(zip(s1, s2)), total=len(s1)):
            remnant = ketju_calculate_bh_merger_remnant_properties(
                m1=self.bh1.m,
                m2=self.bh2.m,
                s1=ss1,
                s2=ss2,
                x1=self.bh1.x.flatten(),
                x2=self.bh2.x.flatten(),
                v1=self.bh1.v.flatten(),
                v2=self.bh2.v.flatten(),
            )
            vkick[i] = np.linalg.norm(remnant["v"]) / self.escape_vel
        _logger.debug(
            f"{np.sum(np.isnan(vkick)) / len(vkick) * 100:.2f}% of calculations from from the Zlochower Lousto relation are NaN!"
        )
        _OOS["vkick_OOS"] = vkick[~np.isnan(vkick)]
        self.stan_data.update(_OOS)

    def set_stan_data(self):
        """
        Set the Stan data dictionary used for sampling.
        """
        self.stan_data = dict(
            N_tot=self.num_obs_collapsed,
            vkick=self.obs_collapsed["vkick"],
            rb=self.obs_collapsed["rb"],
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS()

    def sample_model(self, sample_kwargs=..., diagnose=True):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS
        from previous sample.
        """
        super().sample_model(sample_kwargs, diagnose=diagnose)
        if self._loaded_from_file:
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()

    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        v = super().sample_generated_quantity(gq, force_resample, state)
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state, collapsed=True)
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
        fig, ax = plt.subplots(len(self.latent_qtys), 1, figsize=figsize)
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
            plotting axis of corner plots
        """
        # diagnostic plots
        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent
        )

        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(
            np.quantile(self.obs_collapsed["rb"], 0.1),
            np.quantile(self.obs_collapsed["rb"], 0.9),
        )
        ax.set_xlabel(r"$v/v_\mathrm{esc}$")
        ax.set_ylabel(r"$r_\mathrm{b}/r_{\mathrm{b},0}$")
        self.plot_predictive(
            xmodel="vkick",
            ymodel=self.folded_qtys_posterior[0],
            xobs="vkick",
            yobs=self.folded_qtys[0],
            ax=ax,
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
        Posterior plots for out of sample points.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        ax = self.parameter_corner_plot(
            self.latent_qtys_posterior,
            figsize=figsize,
            labeller=self._labeller_latent_posterior,
        )
        fig = ax.flatten()[0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"corner_OOS_{self._parameter_corner_plot_counter}"
            ),
            fig=fig,
        )

        # out of sample posterior
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(r"$v/v_\mathrm{esc}$")
        ax.set_ylabel(r"$r_\mathrm{b}/r_{\mathrm{b},0}$")
        self.posterior_OOS_plot(
            xmodel="vkick_OOS", ymodel=self.folded_qtys_posterior[0], ax=ax
        )

        # distribution of core radius
        ax = self.plot_generated_quantity_dist(
            ["rb_posterior"],
            state="OOS",
            xlabels=self._folded_qtys_labs[0],
            save=False,
        )
        rb_mode = self.calculate_mode("rb_posterior")
        _logger.info(
            f"Forward-folded core radius mode is {rb_mode*self._rb0:.3f} kpc ({rb_mode:.3f} rb0)"
        )
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        ax.flatten()[0].tick_params(axis="x", which="both", top=False)
        rb02kpc = lambda x: x * self._rb0
        kpc2rb0 = lambda x: x / self._rb0
        secax = ax.flatten()[0].secondary_xaxis("top", functions=(rb02kpc, kpc2rb0))
        secax.set_xlabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, "gqs"), fig=fig)

    def all_plots(self, figsize=None):
        self.plot_latent_distributions(figsize)
        self.all_posterior_pred_plots(figsize)
        self.all_posterior_OOS_plots(figsize)

    @classmethod
    def load_fit(
        cls,
        model_file,
        fit_files,
        figname_base,
        escape_vel,
        premerger_ketjufile,
        rng=None,
    ):
        """
        Restore a stan model from a previously-saved set of csv files

        Parameters
        ----------
        model_file : str
            path to .stan file specifying the likelihood model
        fit_files : str, path-like
            path to previously saved csv files
        figname_base : str
            path-like base name that all plots will share
            _description_
        escape_vel : float
            system escape velocity
        premerger_ketjufile : path-like
            ketju_bhs.hdf5 file of the pre-merger BHs
        rng : np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        C = super().load_fit(model_file, fit_files, figname_base, rng)
        C.escape_vel = escape_vel
        C.premerger_ketjufile = premerger_ketjufile
        return C


class CoreKickExp(_CoreKickBase):
    def __init__(self, model_file, prior_file, figname_base, escape_vel=None, premerger_ketjufile=None, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, escape_vel, premerger_ketjufile, rng)

    @property
    def latent_qtys(self):
        return ["K", "b", "err"]

    @property
    def _latent_qtys_labs(self):
        return [r"$K$", r"$\beta$", r"$\tau$"]


class CoreKickLinear(_CoreKickBase):
    def __init__(self, model_file, prior_file, figname_base, escape_vel=None, premerger_ketjufile=None, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, escape_vel, premerger_ketjufile, rng)

    @property
    def latent_qtys(self):
        return ["a", "b", "err"]

    @property
    def _latent_qtys_labs(self):
        return [r"$a$", r"$b$", r"$\tau$"]


class CoreKickSigmoid(_CoreKickBase):
    def __init__(self, model_file, prior_file, figname_base, escape_vel=None, premerger_ketjufile=None, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, escape_vel, premerger_ketjufile, rng)

    @property
    def latent_qtys(self):
        return ["K", "b", "c", "err"]

    @property
    def _latent_qtys_labs(self):
        return [r"$K$", r"$\beta$", r"$c$", r"$\tau$"]

