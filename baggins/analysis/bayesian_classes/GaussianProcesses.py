from abc import abstractmethod
import os.path
import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.env_config import _cmlogger, baggins_dir
from baggins.plotting import savefig
from baggins.utils import save_data, get_files_in_dir


__all__ = ["_GPBase", "GeneralGP"]

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
        _logger.debug(f"Setting {self.stan_data['N1']} training points")

    def sample_model(self, sample_kwargs={}, diagnose=True):
        super().sample_model(
            sample_kwargs=sample_kwargs, diagnose=diagnose, pathfinder=False
        )
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

    def save_gp_for_plots(self, fname, xkey="x", ykey="y"):
        """
        Save GP data for later plotting

        Parameters
        ----------
        fname : str, pathlike
            file to save data to
        xkey : str, optional
            key for x data, by default "x"
        ykey : str, optional
            key for y data, by default "y"
        """
        data = {
            f"{xkey}": self.stan_data["x1"],
            f"{ykey}": self.sample_generated_quantity(
                self.folded_qtys_posterior[0], state="OOS"
            ),
        }
        save_data(data, fname)


class GeneralGP(_GPBase):
    def __init__(self, figname_base, rng):
        """
        General purpose GP that allows fits a regression to some data stored as a text file.

        Parameters
        ----------
        figname_base : str
            path-like base name that all plots will share
        rng : np.random.Generator
            random number generator, by default None (creates a new instance)
        """
        super().__init__(
            model_file=get_stan_file("gp_analytic"),
            prior_file="",
            figname_base=figname_base,
            rng=rng,
        )
        self._input_qtys_labs = [r"$x$"]
        self._folded_qtys_labs = [r"$y$"]

    @property
    def input_qtys_labs(self):
        return self._input_qtys_labs

    @property
    def folded_qtys_labs(self):
        return self._folded_qtys_labs

    def extract_data(self, d=None, skiprows=0, logx=False, logy=False):
        """
        Read data in from txt file.

        Parameters
        ----------
        d : str, path-like, optional
            file to read, by default None
        skiprows : int, optional
            rows to skip, by default 0
        logx : bool, optional
            fit x in log10 space, by default False
        logy : bool, optional
            fit y in log10 space, by default False

        Raises
        ------
        RuntimeError
            if non-txt file supplied
        """
        d = self._get_data_dir(d)
        try:
            fnames = get_files_in_dir(d, ext=".txt")
        except NotADirectoryError:
            # the individual file names are saved to the input_data_*.yml file
            _ext = os.path.splitext(d)[-1]
            _logger.debug(f"Loading from {_ext} file")
            if _ext == ".yml":
                fnames = d
            elif _ext == ".txt":
                fnames = [d]
            else:
                raise RuntimeError(f"Unknown file type {_ext}")
        except TypeError:
            _logger.debug("TypeError -> taking the first instance")
            fnames = d[0]
        obs = {"x": [], "y": []}
        _logger.debug(f"Files to load {fnames}")
        for f in fnames:
            _logger.info(f"Loading file: {f}")
            _dat = np.loadtxt(f, skiprows=skiprows)
            if _dat.shape[0] == 2 and _dat.shape[1] != 2:
                # convert to column-major
                _dat = _dat.T
            _logger.debug(f"Input data has shape {_dat.shape}")
            # TODO check for 2x2 case
            if logx:
                obs["x"].append(np.log10(_dat[:, 0]))
            else:
                obs["x"].append(_dat[:, 0])
            if logy:
                obs["y"].append(np.log10(_dat[:, 1]))
            else:
                obs["y"].append(_dat[:, 1])

            if not self._loaded_from_file:
                self._add_input_data_file(f)
        self.obs = obs
        self.collapse_observations(["x", "y"])

    def _set_stan_data_OOS(self, N=None):
        """
        Set the out-of-sample Stan data variables.
        Parameters
        ----------
        N : int, optional
            number of OOS points, by default None
        """
        if N is None:
            N = max([len(x) for x in self.obs["x"]]) * 10
        self._num_OOS = N
        xmin = min([np.min(x) for x in self.obs["x"]])
        xmax = max([np.max(x) for x in self.obs["x"]])
        x2 = np.linspace(xmin, xmax, self.num_OOS)
        self.stan_data.update({"x2": x2, "N2": self.num_OOS})

    def set_stan_data(self, *pars):
        """
        Set the data for Stan
        """
        super().set_stan_data(*pars)
        self.stan_data.update(
            {"x1": self.obs_collapsed["x"], "y1": self.obs_collapsed["y"]}
        )

    def posterior_OOS_plot(self, figsize=None):
        """
        Plots for posterior.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axes
        """
        ylims = (
            np.quantile(self.obs_collapsed["y"], 0.01),
            np.quantile(self.obs_collapsed["y"], 0.99),
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.add_data_to_predictive_plot(ax=ax, xobs="x", yobs="y")
        ax.set_ylim(*ylims)
        ax.set_xlabel(self._input_qtys_labs[0])
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.posterior_OOS_plot(
            xmodel="x2", ymodel=self.folded_qtys_posterior[0], ax=ax, smooth=True
        )
        return ax
