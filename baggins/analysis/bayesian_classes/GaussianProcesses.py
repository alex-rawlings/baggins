from abc import abstractmethod
import os.path
import numpy as np
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.env_config import _cmlogger, baggins_dir
from baggins.plotting import savefig, get_all_axes_from_plot_collection
from baggins.utils import save_data


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
        self._independent_qtys = ["x"]
        self._independent_qtys_OOS = ["x_OOS"]
        self.independent_qtys_labs = [r"$x$"]
        self._dependent_qtys = ["y"]
        self._dependent_qtys_posterior = [
            f"{v}_posterior" for v in self._dependent_qtys
        ]
        self._dependent_qtys_prior = [f"{v}_prior" for v in self._dependent_qtys]
        self._dependent_qtys_OOS = [f"{v}_OOS" for v in self._dependent_qtys]
        self.dependent_qtys_labs = [r"$y$"]
        self._make_xy_labellers()
        self._latent_qtys = ["rho", "alpha", "err"]
        self._latent_qtys_labs = [r"$\rho$", r"$\alpha$", r"$\tau$"]
        self._make_latent_labellers()

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

    # ----------------------------------------------------------------------
    # Abstract methods
    # ----------------------------------------------------------------------

    @abstractmethod
    def extract_data(self):
        return super().extract_data()

    @abstractmethod
    def _set_stan_data_OOS(self, N):
        return super()._set_stan_data_OOS(N)

    # ----------------------------------------------------------------------
    # Stan Data
    # ----------------------------------------------------------------------

    def set_stan_data(self, *kwargs):
        if self.stan_data is None:
            self.stan_data = {}
        self.stan_data.update(
            {
                "N_obs": self.num_obs_collapsed,
                self._independent_qtys[0]: self.obs_collapsed[
                    self._independent_qtys[0]
                ],
                self._dependent_qtys[0]: self.obs_collapsed[self._dependent_qtys[0]],
            }
        )
        self._set_stan_data_OOS(**kwargs)
        _logger.debug(f"Setting {self.stan_data['N_obs']} training points")

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------

    def sample_model(self, sample_kwargs={}, diagnose=True):
        super().sample_model(
            sample_kwargs=sample_kwargs, diagnose=diagnose, pathfinder=False
        )
        if self._loaded_from_file:
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()

    def diagnose_sample(self):
        return super().diagnose_sample(self.latent_qtys)

    # ----------------------------------------------------------------------
    # Plotting methods
    # ----------------------------------------------------------------------

    def plot_latent_distributions(self, save=True):
        """
        Plot distributions of the latent parameters of the model.

        Parameters
        ----------
        save : bool, optional
            save the figure, by default True

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
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
        return ax

    def plot_posterior_predictive(self, save=True, **kwargs):
        """
        Plot posterior predictive regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        ax : matplotlib.Axes.axes
            plotting axes
        """
        pc = super().plot_posterior_predictive(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_postpred_plot_name))
        return ax

    def plot_prior_predictive(self, save=True, **kwargs):
        """
        Plot prior predictive regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        ax : matplotlib.Axes.axes
            plotting axes
        """
        pc = super().plot_prior_predictive(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_priorpred_plot_name))
        return ax

    def plot_posterior_OOS(self, save=True, **kwargs):
        """
        Plot posterior out-of-sample regression model.

        Parameters
        ----------
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        ax : matplotlib.Axes.axes
            plotting axes
        """
        pc = super().plot_posterior_OOS(**kwargs)
        ax = pc.get_viz("plot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if save:
            savefig(next(self.gen_postOOS_plot_name))
        return ax

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

        # transformed latent parameter distributions
        pc = self.parameter_corner_plot(
            self.latent_qtys_posterior,
            figsize=(len(self.latent_qtys_posterior), len(self.latent_qtys_posterior)),
            labeller=self._labeller_latent_posterior,
        )
        fig = pc.get_viz("figure")
        fig.suptitle("Latent parameters (out-sample)")
        savefig(next(self.gen_corner_plot_name), fig=fig)

        self.parameter_diagnostic_plots(
            self.latent_qtys, labeller=self._labeller_latent, figsize=figsize
        )

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
            f"{xkey}": self.access_independent_qty(self._independent_qtys[0]),
            f"{ykey}": self.sample_generated_quantity(
                self.sample_generated_quantity(self.dependent_qtys[0])
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
        obs = {"x": [], "y": []}
        d = self._get_data_files(d)
        if self._loaded_from_file:
            fname = d[0]
            skiprows = self._input_data_and_pars["data_opts"]["skiprows"]
            logx = self._input_data_and_pars["data_opts"]["logx"]
            logy = self._input_data_and_pars["data_opts"]["logx"]
        else:
            fname = d
            self._input_data_and_pars["data_opts"] = dict(
                skiprows=skiprows, logx=logx, logy=logy
            )
        _logger.info(f"Loading file: {fname}")
        _dat = np.loadtxt(fname, skiprows=skiprows)
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
            self._add_input_data_file(fname)
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
        OOS_data = super()._set_stan_data_OOS(N)
        xmin = min([np.min(x) for x in self.obs["x"]])
        xmax = max([np.max(x) for x in self.obs["x"]])
        x = np.linspace(xmin, xmax, self.num_OOS)
        OOS_data.update({self._independent_qtys_OOS[0]: x})
        self.stan_data.update(OOS_data)
