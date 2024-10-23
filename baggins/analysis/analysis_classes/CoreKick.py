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
        self._restrict_vel = False
        self._vmax = None

    @property
    def folded_qtys(self):
        return self._folded_qtys

    @property
    def folded_qtys_posterior(self):
        return self._folded_qtys_posterior

    @property
    def vmax(self):
        return self._vmax

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
        return MapLabeller(
            dict(zip(self._latent_qtys_posterior, self._latent_qtys_labs))
        )

    @property
    def rb0(self):
        return self._rb0

    def extract_data(self, d=None, pars=None):
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

    def _set_stan_data_OOS(self, N=10000):
        """
        Set the OOS data points for the Stan model

        Parameters
        ----------
        N : int, optional
            number of recoil velocity samples to draw, by default 10000
        """

        def _spin_setter(nn):
            t, p = uniform_sample_sphere(nn * 2, rng=self._rng)
            spin_mag = scipy.stats.beta.rvs(
                *zlochower_dry_spins.values(),
                random_state=self._rng,
                size=nn * 2,
            )
            spins = convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
            return spins[:nn, :], spins[nn:, :]

        _OOS = {"N_OOS": N}
        self._num_OOS = _OOS["N_OOS"]
        self._vmax = np.nanmax(self.obs_collapsed["vkick"])
        vkick = np.full(_OOS["N_OOS"], np.nan)
        _logger.debug(f"Maximum fitted velocity is {self.vmax:.2e}")
        remaining = np.ones(N, dtype=bool)
        iters = 0
        max_iters = 100
        while np.any(remaining) and iters < max_iters:
            # generate spins
            s1, s2 = _spin_setter(np.sum(remaining))
            update_idxs = np.where(remaining == 1)[0]
            for i, (ss1, ss2) in tqdm(
                enumerate(zip(s1, s2)),
                total=len(s1),
                desc=f"Sampling BH spins (iteration {iters})",
            ):
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
                vkick[update_idxs[i]] = np.linalg.norm(remnant["v"]) / self.escape_vel
            remaining = np.isnan(vkick)
            if self._restrict_vel:
                remaining = np.logical_or(remaining, vkick > self.vmax)
            _logger.debug(
                f"Completed iteration {iters}, but there are {np.sum(remaining)} kick values that need to be resampled"
            )
            iters += 1
        _logger.debug(
            f"{np.sum(vkick > self.vmax)} kick velocities are larger than the maximum value of vkick used in model constraint"
        )
        if iters >= max_iters:
            _logger.error(
                f"The number of iterations in sampling the BH spins has exceeded the maximum number of {max_iters} without converging!"
            )
        vkick.sort()
        assert np.all(np.sign(np.diff(vkick)) > 0)
        _OOS["vkick_OOS"] = vkick
        _logger.debug(f"Length of OOS vkick: {len(_OOS['vkick_OOS'])}")
        self.stan_data.update(_OOS)

    def set_stan_data(self, restrict_v=False):
        """
        Set the Stan data dictionary used for sampling.

        Parameters
        ----------
        restrict_v : bool, optional
            restrict the sampled recoil velocity values to less than the
            maximum velocity used in the fitting, by default False
        """
        self._restrict_vel = restrict_v
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
            xlabels=self._folded_qtys_labs,
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
    def __init__(
        self,
        model_file,
        prior_file,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(
            model_file, prior_file, figname_base, escape_vel, premerger_ketjufile, rng
        )

    @property
    def latent_qtys(self):
        return ["K", "b", "err"]

    @property
    def _latent_qtys_labs(self):
        return [r"$K$", r"$\beta$", r"$\tau$"]


class CoreKickLinear(_CoreKickBase):
    def __init__(
        self,
        model_file,
        prior_file,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(
            model_file, prior_file, figname_base, escape_vel, premerger_ketjufile, rng
        )

    @property
    def latent_qtys(self):
        return ["a", "err"]

    @property
    def _latent_qtys_labs(self):
        return [r"$a$", r"$\tau$"]


class CoreKickSigmoid(_CoreKickBase):
    def __init__(
        self,
        model_file,
        prior_file,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(
            model_file, prior_file, figname_base, escape_vel, premerger_ketjufile, rng
        )

    @property
    def latent_qtys(self):
        return ["K", "b", "err"]

    @property
    def _latent_qtys_labs(self):
        return [r"$K$", r"$\beta$", r"$\tau$"]
