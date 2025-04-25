from abc import abstractmethod
import os.path
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.env_config import _cmlogger, baggins_dir
from baggins.plotting import savefig
from baggins.utils import save_data


__all__ = ["_GPBase"]

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

    def sample_model(self, sample_kwargs=..., diagnose=True):
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
