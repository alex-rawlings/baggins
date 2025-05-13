import os.path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
from arviz import plot_dist, plot_hdi
from baggins.analysis.bayesian_classes.GaussianProcesses import _GPBase, get_stan_file
from baggins.analysis.analyse_ketju import get_bound_binary
from baggins.env_config import _cmlogger
from baggins.general.units import kpc
from baggins.literature import SMBHSpins, ketju_calculate_bh_merger_remnant_properties
from baggins.plotting import savefig
from baggins.utils import get_ketjubhs_in_dir, get_files_in_dir

__all__ = ["VkickApocentreGP"]

_logger = _cmlogger.getChild(__name__)


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
        self._input_qtys_labs = [r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$"]
        self._folded_qtys_labs = [r"$r_\mathrm{apo}/\mathrm{kpc}$"]
        self.premerger_ketjufile = premerger_ketjufile
        self.bh1 = None
        self.bh2 = None
        self._num_OOS = 2000
        # some OOS samples will be dropped if they are a kick velocity above
        # the maximum vkick from the data
        self._num_OOS_requested = self._num_OOS

    @property
    def input_qtys_labs(self):
        return self._input_qtys_labs

    @property
    def folded_qtys_labs(self):
        return self._folded_qtys_labs

    def extract_data(self, d=None, minvel=None, maxvel=None):
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
        obs = {"vkick": [], "rapo": [], "tapo": []}
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            # get kick velocity from file name
            _v = float(
                os.path.splitext(os.path.basename(f))[0].replace("kick-vel-", "")
            )
            # handle the case of 0 km/s
            if _v < 1e-12:
                _v = 1
            if maxvel is not None and _v > maxvel:
                continue
            if minvel is not None and _v < minvel:
                continue
            # load data from file
            if _v <= 1:
                # for very low vkick, we have ~0 displacement
                obs["rapo"].append([1e-3])
                obs["tapo"].append([1e-3])
            else:
                # XXX skip the first few snapshots, in most use cases we expect
                # there to be many more than just 3 snapshots before apocentre
                # anyway
                _dat = np.loadtxt(f, skiprows=1)[3:, :2]
                _t = _dat[:, 0]
                _r = _dat[:, 1]
                if np.any(np.diff(_r) < 0):
                    # we have an instance where the distance of the BH to
                    # centre is decreasing
                    obs["rapo"].append([np.nanmax(_r)])
                    obs["tapo"].append([_t[np.argmax(_r)] * 1e3])  # convert to Myr
                else:
                    _logger.warning(
                        f"Velocity {_v} km/s did not reach an apocentre! Skipping"
                    )
                    continue
            obs["vkick"].append([_v])
            # track this file on the input data list
            if not self._loaded_from_file:
                self._add_input_data_file(f)
        self.obs = obs
        self.transform_obs("vkick", "log10_vkick", lambda x: np.log10(x))
        self.transform_obs("rapo", "log10_rapo", lambda x: np.log10(x))
        self.collapse_observations(
            ["vkick", "rapo", "log10_vkick", "log10_rapo", "tapo"]
        )
        # extract BH data at the timestep before merger
        kfile = get_ketjubhs_in_dir(self.premerger_ketjufile)[0]
        bh1, bh2, *_ = get_bound_binary(kfile)
        self.bh1 = bh1[-1]
        self.bh2 = bh2[-1]

    def _ketju_particle_to_gadget_units(self):
        """
        Move to Gadget units: kpc, km/s, 1e10Msol
        """
        self.bh1.x /= kpc
        self.bh2.x /= kpc
        self.bh1.v /= ketjugw.units.km_per_s
        self.bh2.v /= ketjugw.units.km_per_s
        self.bh1.m /= 1e10
        self.bh2.m /= 1e10

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
            spins = SMBHSpins("zlochower_dry", "skewed")
            L = ketjugw.orbital_angular_momentum(self.bh1, self.bh2).flatten()
            s1 = spins.sample_spins(self.num_OOS, L=L)
            s2 = spins.sample_spins(self.num_OOS, L=L)
            vkick = np.full(self.num_OOS, np.nan)
            self._ketju_particle_to_gadget_units()
            for i, (ss1, ss2) in tqdm(
                enumerate(zip(s1, s2)), total=len(s1), desc="Sampling kicks"
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
                vkick[i] = np.linalg.norm(remnant["v"])
            _logger.debug(
                f"{np.sum(np.isnan(vkick)) / len(vkick) * 100:.2f}% of calculations from from the Zlochower Lousto relation are NaN!"
            )
            _OOS["x2"] = vkick[
                np.logical_and(
                    ~np.isnan(vkick), vkick < np.max(np.array(self.obs["vkick"]))
                )
            ]
            try:
                assert len(_OOS["x2"]) > 1
            except AssertionError:
                _logger.exception(
                    "At least two points are required for GP interpolation!",
                    exc_info=True,
                )
                raise
            self._num_OOS = len(_OOS["x2"])
            _OOS["N2"] = self.num_OOS
        else:
            try:
                assert isinstance(vkickOOS, np.ndarray)
                vkickOOS = vkickOOS.flatten()
            except AssertionError:
                _logger.exception(
                    f"User-defined OOS kick velocities must be an array, not {type(vkickOOS)}",
                    exc_info=True,
                )
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
        """
        All standard plots for a model

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        super().all_plots(figsize)
        ylims = (
            np.quantile(self.obs_collapsed["rapo"], 0.01),
            np.quantile(self.obs_collapsed["rapo"], 0.99),
        )
        # posterior predictive check
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_yscale("log")
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
        ax.set_yscale("log")
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )

        rapo_mode = self.calculate_mode("y")
        _logger.info(f"Forward-folded apocentre mode is {rapo_mode:.3f} kpc")

        # marginal distribution of dependent variable
        fig, ax_rapo = plt.subplots()
        # add a secondary axis, turning off ticks from the top axis (if they are there)
        ax_rapo.tick_params(axis="x", which="both", top=False)
        self.plot_generated_quantity_dist(
            ["y"],
            bounds=[(0, 1e4)],
            state="OOS",
            xlabels=self._folded_qtys_labs,
            ax=ax_rapo,
        )

    def plot_kick_distribution(self, ax=None, save=True, **kwargs):
        """
        Plot sampled kick velocity distribution.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            plotting axes, by default None
        save : bool, optional
            save figure, by default True

        Returns
        -------
        ax : matplotlib.Axes, optional
            plotting axes, by default None
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(self.input_qtys_labs[0])
            ax.set_ylabel(r"$P(v_\mathrm{kick}\cos(\theta))$")
        else:
            fig = ax.get_figure()
        plot_dist(self.stan_data["x2"], **kwargs)
        if save:
            savefig(
                self._make_fig_name(
                    self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"
                ),
                fig=fig,
            )
            self._gq_distribution_plot_counter += 1
        return ax

    def fraction_apo_above_threshold(self, threshold, proj=False):
        """
        Determine the fraction of apocentres above some distance threshold
        given that the SMBH has a reasonable amount of mass bound to it.

        Parameters
        ----------
        threshold : callable
            distance threshold function
        proj : bool, optional
            use a projected distance, by default False

        Returns
        -------
        : float
            fraction of apocentres above threshold
        """
        r_apo = self.sample_generated_quantity("y", state="OOS")
        # make sure there are no negative values
        mask = r_apo >= 0
        if proj:
            r_apo = r_apo * np.sin(np.arccos(self._rng.uniform(size=r_apo.shape)))
        # fraction above threshold
        # relative to the total kick distribution, i.e. not truncated to some
        # upper value
        vk = np.tile(self.stan_data["x2"], mask.shape[0]).reshape(mask.shape)
        return np.nansum(r_apo[mask] > threshold(vk)[mask]) / (
            self._num_OOS_requested * mask.shape[0]
        )

    def angle_to_exceed_threshold(self, threshold):
        """
        Determine the angle to exceed some threshold distance

        Parameters
        ----------
        threshold : callable
            distance threshold function

        Returns
        -------
        : array-like
            minimum angle
        """
        r_apo = self.sample_generated_quantity("y", state="OOS")
        vk = np.tile(self.stan_data["x2"], r_apo.shape[0]).reshape(r_apo.shape)
        theta = np.arcsin(threshold(vk) / r_apo) * 180 / np.pi  # in degrees
        # set apocentres below threshold to nan
        theta[r_apo < threshold(vk)] = np.nan
        return theta

    def plot_angle_to_exceed_threshold(
        self, threshold, levels=None, ax=None, save=True, smooth_kwargs=None
    ):
        """
        Plot the minimum angle to exceed a distance threshold as a function of kick velocity.

        Parameters
        ----------
        threshold : callable
            distance threshold function the BH must exceed
        ax : matplotlib.Axes, optional
            plotting axes, by default None
        save : bool, optional
            save the plot, by default True
        smooth_kwargs : dict, by default None
            smoothing parameters parsed to az.plot_hdi()

        Returns
        -------
        ax : pyplot.Axes
            plotting axes
        """
        theta = self.angle_to_exceed_threshold(threshold=threshold)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(self.input_qtys_labs[0])
            ax.set_ylabel(r"$\theta$")
        else:
            fig = ax.get_figure()
        if levels is None:
            levels = self._default_hdi_levels
        levels.sort(reverse=True)
        cmapper, sm = self._make_default_hdi_colours(levels)
        for lev in levels:
            _logger.debug(f"Fitting level {lev}")
            plot_hdi(
                self.stan_data["x2"],
                theta,
                hdi_prob=lev / 100,
                ax=ax,
                plot_kwargs={"c": cmapper(lev)},
                fill_kwargs={
                    "color": cmapper(lev),
                    "alpha": 0.8,
                    "label": f"{lev}% HDI",
                    "edgecolor": None,
                },
                smooth=True,
                smooth_kwargs=smooth_kwargs,
                hdi_kwargs={"skipna": True},
            )
        if save:
            savefig(
                self._make_fig_name(
                    self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"
                ),
                fig=fig,
            )
            self._gq_distribution_plot_counter += 1
        return ax

    def plot_observable_fraction(
        self, threshold, bins=None, ax=None, cols=None, save=True, **kwargs
    ):
        """
        Plot observability probability of sampled kick velocity distribution.

        Parameters
        ----------
        threshold : callable
            distance threshold function the BH must exceed
        min_v : float, optional
            lower cut for velocity, by default None
        n_bootstrap : int, optional
            number of ECDF bootstrap samples, by default 100
        ax : matplotlib.Axes, optional
            plotting axes, by default None
        save : bool, optional
            save figure, by default True

        Returns
        -------
        ax : matplotlib.Axes, optional
            plotting axes, by default None
        """
        theta = self.angle_to_exceed_threshold(threshold=threshold)
        draws = theta.shape[0]
        vk = np.tile(self.stan_data["x2"], draws).reshape(theta.shape)
        theta = theta.flatten()
        vk = vk.flatten()
        # -> visible = 1 - not_visible
        # ->         = 1 - 2 * 2pi(1-cosT) / 4pi
        # ->         = cosT
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(self.input_qtys_labs[0])
            ax.set_ylabel(r"$f(v_\mathrm{kick}\cos(\theta))$")
        else:
            fig = ax.get_figure()
        weights = np.cos(theta * np.pi / 180)
        weights[np.isnan(weights)] = 0
        N = draws * self._num_OOS_requested
        if cols is None:
            cols = [None, None]
        ax.hist(
            vk,
            bins=bins,
            density=False,
            weights=np.ones_like(vk) / N,
            label=r"$\mathrm{SMBH\;recoil}$",
            color=cols[0],
            **kwargs,
        )
        h = ax.hist(
            vk,
            bins=bins,
            density=False,
            weights=weights / N,
            label=r"$\mathrm{BRC\; not\; occulted}$",
            color=cols[1],
            **kwargs,
        )
        if kwargs.pop("cumulative", False):
            _logger.debug(f"Final bin has (cumulative) value {h[0][-1]:.3e}")

        if save:
            ax.legend()
            savefig(
                self._make_fig_name(
                    self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"
                ),
                fig=fig,
            )
            self._gq_distribution_plot_counter += 1
        return ax

    def _interpolate_apo_to_time(self, r):
        """
        Interpolate apocentre distance to apocentre time.

        Parameters
        ----------
        r : array-like
            apocentres to sample

        Returns
        -------
        : array-like
            interpolated apocentre times
        """
        x = self.obs_collapsed["rapo"]
        y = self.obs_collapsed["tapo"]
        return np.interp(r, x, y)

    def plot_apocentre_time_distribution(
        self, ax=None, save=True, cumulative=False, **kwargs
    ):
        """
        Plot the apocentre time distribution

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            plotting axes, by default None
        save : bool, optional
            save plot, by default True
        cumulative : bool, optional
            plot cumulative distribution, by default False

        Returns
        -------
        ax : matplotlib.Axes, optional
            plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel(r"$t_\mathrm{apo}/\mathrm{Myr}$")
        ax.set_ylabel(r"$\mathrm{CDF}$" if cumulative else r"$\mathrm{PDF}$")
        r_apo = self.sample_generated_quantity("y", state="OOS")
        t_apo = self._interpolate_apo_to_time(r_apo)
        plot_dist(t_apo, ax=ax, cumulative=cumulative, **kwargs)
        if save:
            savefig(
                self._make_fig_name(
                    self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"
                ),
                fig=fig,
            )
            self._gq_distribution_plot_counter += 1
        return ax

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
