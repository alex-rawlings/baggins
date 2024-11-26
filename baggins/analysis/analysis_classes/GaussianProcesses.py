from abc import abstractmethod
import os.path
import numpy as np
import scipy.stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from ketjugw.units import km_per_s
from baggins.analysis.analysis_classes.StanModel import HierarchicalModel_2D
from baggins.analysis.analyse_ketju import get_bound_binary
from baggins.env_config import _cmlogger, baggins_dir
from baggins.general.units import kpc
from baggins.literature import (
    zlochower_dry_spins,
    ketju_calculate_bh_merger_remnant_properties,
)
from baggins.mathematics import uniform_sample_sphere, convert_spherical_to_cartesian
from baggins.plotting import savefig
from baggins.utils import save_data, load_data, get_ketjubhs_in_dir, get_files_in_dir


__all__ = ["_GPBase", "VkickCoreradiusGP", "CoreradiusVkickGP", "VkickApocentreGP"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/gaussian-process/{f.rstrip('.stan')}.stan")


class _GPBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng) -> None:
        """
        Base class for Gaussian processes.

        Parameters
        ----------
        See input to HierarchicalModel_2D.
        Note that the class requires an RNG object to be given, as OOS
        quantities are fit in the model() section of the Stan code, making the
        model unable to be run for differing inputs when loading from a set of
        saved .csv files.
        """
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["rho", "alpha", "sigma"]
        self._latent_qtys_labs = [r"$\rho$", r"$\alpha$", r"$\sigma$"]
        self._labeller_latent = MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._folded_qtys = ["y1"]
        self._folded_qtys_posterior = ["y"]

    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def folded_qtys(self):
        return self._folded_qtys

    @property
    def folded_qtys_posterior(self):
        return self._folded_qtys_posterior

    @abstractmethod
    def extract_data(self):
        return super().extract_data()

    @abstractmethod
    def _set_stan_data_OOS(self):
        return super()._set_stan_data_OOS()

    @abstractmethod
    def set_stan_data(self, *pars):
        self.stan_data = dict(
            N1=self.num_obs_collapsed,
        )
        if not self._loaded_from_file:
            self._set_stan_data_OOS(*pars)

    def sample_model(self, sample_kwargs=..., diagnose=True):
        super().sample_model(sample_kwargs=sample_kwargs, diagnose=diagnose, pathfinder=False)
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
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        self.plot_generated_quantity_dist(
            self._latent_qtys, ax=ax, xlabels=self._latent_qtys_labs
        )
        return ax

    def diag_plots(self, figsize=None):
        """
        Plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        type_str = "prior" if self._fit is None else "posterior"

        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent, figsize=figsize
        )

        # latent quantities
        self.plot_latent_distributions(figsize=figsize)
        ax1 = self.parameter_corner_plot(
            self.latent_qtys,
            figsize=figsize,
            labeller=self._labeller_latent,
            combine_dims={"dim"},
        )
        fig1 = ax1[0, 0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base,
                f"corner_{type_str}_{self._parameter_corner_plot_counter}",
            ),
            fig=fig1,
        )

    @abstractmethod
    def all_plots(self, figsize=None):
        self.diag_plots(figsize=figsize)
        pass

    def save_gp_for_plots(self, fname):
        data = dict(
            vkick=self.stan_data["x1"],
            rb=self.sample_generated_quantity(
                self.folded_qtys_posterior[0], state="OOS"
            ),
        )
        save_data(data, fname)


class VkickCoreradiusGP(_GPBase):
    def __init__(
        self,
        figname_base,
        escape_vel=None,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qtys_labs = [r"$v/v_\mathrm{esc}$"]
        self._folded_qtys_labs = [r"$r_\mathrm{b}/r_{\mathrm{b},0}$"]
        self.escape_vel = escape_vel
        self.premerger_ketjufile = premerger_ketjufile
        self.bh1 = None
        self.bh2 = None
        self._rb0 = np.nan

    def extract_data(self, d=None):
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
            obs["vkick"].append([float(k) / self.escape_vel])
            obs["rb"].append([np.nanmedian(v.flatten() / rb0.flatten())])
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
        _OOS = {"N2": 1000}
        self._num_OOS = _OOS["N2"]
        t, p = uniform_sample_sphere(_OOS["N2"] * 2, rng=self._rng)
        spin_mag = scipy.stats.beta.rvs(
            *zlochower_dry_spins.values(),
            random_state=self._rng,
            size=_OOS["N2"] * 2,
        )
        spins = convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
        s1 = spins[: _OOS["N2"], :]
        s2 = spins[_OOS["N2"] :, :]
        vkick = np.full(_OOS["N2"], np.nan)
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
        _OOS["x2"] = vkick[~np.isnan(vkick)]
        self.stan_data.update(_OOS)

    def set_stan_data(self):
        super().set_stan_data()
        self.stan_data.update(
            dict(x1=self.obs_collapsed["vkick"], y1=self.obs_collapsed["rb"])
        )

    def all_plots(self, figsize=None):
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["rb"], 0.01),
            np.quantile(self.obs_collapsed["rb"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="x1",
            ymodel=self.folded_qtys_posterior[0],
            xobs="vkick",
            yobs="rb",
            ax=ax,
        )

        # OOS
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

        rb_mode = self.calculate_mode("y")
        _logger.info(
            f"Forward-folded core radius mode is {rb_mode*self._rb0:.3f} kpc ({rb_mode:.3f} rb0)"
        )

        # marginal distribution of dependent variable
        fig, ax_rb = plt.subplots()
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        ax_rb.tick_params(axis="x", which="both", top=False)
        rb02kpc = lambda x: x * self._rb0
        kpc2rb0 = lambda x: x / self._rb0
        secax = ax_rb.secondary_xaxis("top", functions=(rb02kpc, kpc2rb0))
        secax.set_xlabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
        self.plot_generated_quantity_dist(
            ["y"],
            state="OOS",
            bounds=[(0, 3)],
            xlabels=self._folded_qtys_labs,
            ax=ax_rb,
        )

    @classmethod
    def load_fit(
        cls,
        model_file,
        fit_files,
        figname_base,
        escape_vel,
        premerger_ketjufile,
        rng,
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


class CoreradiusVkickGP(_GPBase):
    def __init__(self, figname_base, rng) -> None:
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qtys_labs = [r"$r_\mathrm{b}/r_{\mathrm{b},0}$"]
        self._folded_qtys_labs = [r"$v/v_\mathrm{esc}$"]
        self._rb0 = np.nan

    def extract_data(self, d=None):
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


        Parameters
        ----------
        d : path-like, optional
            file of core radius samples, by default None (paths read from
            `_input_data_files`)
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
            obs["vkick"].append([float(k)])
            # take the median core radius value for each recoil velocity
            obs["rb"].append([np.nanmedian(v.flatten() / rb0.flatten())])
        self._rb0 = np.nanmedian(rb0)
        self.obs = obs
        if not self._loaded_from_file:
            self._add_input_data_file(d)
        self.collapse_observations(["vkick", "rb"])

    def _set_stan_data_OOS(self):
        """
        Set the out-of-sample Stan data variable for core radius normalised to
        rb0 (binary-scoured radius)
        """
        _OOS = {"N2": 1000}
        self._num_OOS = _OOS["N2"]
        _OOS["x2"] = np.linspace(1, np.max(self.obs_collapsed["rb"]), self.num_OOS)
        self.stan_data.update(_OOS)

    def set_stan_data(self):
        super().set_stan_data()
        self.stan_data.update(
            dict(x1=self.obs_collapsed["rb"], y1=self.obs_collapsed["vkick"])
        )

    def all_plots(self, figsize=None):
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["vkick"], 0.01),
            np.quantile(self.obs_collapsed["vkick"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="x1",
            ymodel=self.folded_qtys_posterior[0],
            xobs="rb",
            yobs="vkick",
            ax=ax,
        )

        # OOS
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

        vkick_mode = self.calculate_mode("y")
        _logger.info(f"Forward-folded kick velocity mode is {vkick_mode:.3f} km/s")

        # marginal distribution of dependent variable
        fig, ax_vk = plt.subplots()
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        self.plot_generated_quantity_dist(
            ["y"],
            state="OOS",
            bounds=[(0, 400)],
            xlabels=self._folded_qtys_labs,
            ax=ax_vk,
        )


class VkickApocentreGP(_GPBase):
    def __init__(
        self,
        figname_base,
        premerger_ketjufile=None,
        rng=None,
    ) -> None:
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qtys_labs = [r"$v_\mathrm{kick}$"]
        self._folded_qtys_labs = [r"$r_\mathrm{apo}$"]
        self.premerger_ketjufile = premerger_ketjufile
        self.bh1 = None
        self.bh2 = None
        self._num_OOS = 1000

    def extract_data(self, d=None, maxvel=None):
        """
        Data extraction and manipulation required by the CoreKick model.
        Due to the time complexity of obtaining apocentre information from snapshots, we take Atte's format from the `core-kick` study as default input.
        # TODO maybe we can allow for some other input methods?
        The `pars` parameter is unused and included only for compatability with
        the parent class.


        Parameters
        ----------
        d : path-like, optional
            file of core radius samples, by default None (paths read from
            `_input_data_files`)
        maxvel : float
            maximum velocity to fit to, by default None
        """
        try:
            assert self.premerger_ketjufile is not None
        except AssertionError:
            _logger.exception(
                "Attributes `premerger_ketjufile` must be set before extracting data",
                exc_info=True,
            )
            raise
        d = self._get_data_dir(d)
        try:
            fnames = get_files_in_dir(d, ext=".txt")
        except NotADirectoryError:
            # the individual file names are saved to the input_data_*.yml file
            fnames = d
        except TypeError:
            fnames = d[0]
        obs = {"vkick": [], "rapo": []}
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            # get kick velocity from file name
            _v = float(os.path.splitext(os.path.basename(f))[0].replace("kick-vel-", ""))
            # handle the case of 0 km/s
            if _v < 1e-12:
                _v = 1
            if maxvel is not None and _v > maxvel:
                continue
            obs["vkick"].append(
                [_v]
            )
            # load data from file
            if _v <= 1:
                # for very low vkick, we have ~0 displacement
                obs["rapo"].append([1e-3])
            else:
                obs["rapo"].append(
                    [np.nanmax(np.loadtxt(f, skiprows=1)[:, 1])]
                )
            # track this file on the input data list
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        self.obs = obs
        self.transform_obs("vkick", "log10_vkick", lambda x: np.log10(x))
        self.transform_obs("rapo", "log10_rapo", lambda x: np.log10(x))
        self.collapse_observations(["vkick", "rapo", "log10_vkick", "log10_rapo"])
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

    def _set_stan_data_OOS(self, vkickOOS=None):
        """
        Set the out-of-sample Stan data variables. self.num_OOS points will be
        used.
        BH spins are uniformly sampled on the sphere, with magnitude from the
        Zlochower Lousto "dry" distribution.

        Parameters
        ----------
        vkickOOS : np.array, optional
            desired values of kick velocity to sample, by default None
        """
        _OOS = {"N2": None}
        if vkickOOS is None:
            # randomly sample recoil velocities from Zlochower Lousto
            t, p = uniform_sample_sphere(self.num_OOS * 2, rng=self._rng)
            spin_mag = scipy.stats.beta.rvs(
                *zlochower_dry_spins.values(),
                random_state=self._rng,
                size=self.num_OOS * 2,
            )
            spins = convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
            s1 = spins[: self.num_OOS, :]
            s2 = spins[self.num_OOS :, :]
            vkick = np.full(self.num_OOS, np.nan)
            for i, (ss1, ss2) in tqdm(enumerate(zip(s1, s2)), total=len(s1), desc="Sampling kicks"):
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
                vkick[i] = np.linalg.norm(remnant["v"])
            _logger.debug(
                f"{np.sum(np.isnan(vkick)) / len(vkick) * 100:.2f}% of calculations from from the Zlochower Lousto relation are NaN!"
            )
            _OOS["x2"] = vkick[np.logical_and(
                ~np.isnan(vkick),
                vkick < np.max(np.array(self.obs["vkick"]))
            )]
            try:
                assert len(_OOS["x2"]) > 1
            except AssertionError:
                _logger.exception("At least two points are required for GP interpolation!", exc_info=True)
                raise
            self._num_OOS = len(_OOS["x2"])
            _OOS["N2"] = self.num_OOS
        else:
            try:
                assert isinstance(vkickOOS, np.ndarray)
                vkickOOS = vkickOOS.flatten()
            except AssertionError:
                _logger.exception(f"User-defined OOS kick velocities must be an array, not {type(vkickOOS)}", exc_info=True)
                raise
            _OOS["x2"] = vkickOOS
            _OOS["N2"] = len(_OOS["x2"])
        self.stan_data.update(_OOS)

    def set_stan_data(self, vkickOOS=None):
        super().set_stan_data(vkickOOS)
        self.stan_data.update(
            dict(x1=self.obs_collapsed["vkick"], y1=self.obs_collapsed["rapo"])
        )

    def all_plots(self, figsize=None):
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["rapo"], 0.01),
            np.quantile(self.obs_collapsed["rapo"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="x1",
            ymodel=self.folded_qtys_posterior[0],
            xobs="vkick",
            yobs="rapo",
            ax=ax,
        )

        # OOS
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

        rapo_mode = self.calculate_mode("y")
        _logger.info(
            f"Forward-folded apocentre mode is {rapo_mode:.3f} kpc"
        )

        # marginal distribution of dependent variable
        fig, ax_rapo = plt.subplots()
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        ax_rapo.tick_params(axis="x", which="both", top=False)
        self.plot_generated_quantity_dist(
            ["y"],
            bounds=[(0,1e4)],
            state="OOS",
            xlabels=self._folded_qtys_labs,
            ax=ax_rapo,
        )

    @classmethod
    def load_fit(
        cls,
        fit_files,
        figname_base,
        premerger_ketjufile,
        rng,
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
        premerger_ketjufile : path-like
            ketju_bhs.hdf5 file of the pre-merger BHs
        rng : np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        C = super().load_fit(fit_files, figname_base, rng)
        C.premerger_ketjufile = premerger_ketjufile
        return C
