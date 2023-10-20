import numpy as np
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
from . import HierarchicalModel_2D, HMQuantitiesBinaryData
from ...env_config import _cmlogger

__all__ = ["PQModelSimple", "PQModelHierarchy"]

_logger = _cmlogger.getChild(__name__)

class _PQModelBase(HierarchicalModel_2D):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self._latent_qtys = ["Hps", "K"]
        self._latent_qtys_labs = [r"$H\rho/\sigma$", r"$K$"]
        self._labeller_latent = MapLabeller(dict(zip(self._latent_qtys, self._latent_qtys_labs)))

    @property
    def latent_qtys(self):
        return self._latent_qtys


    def extract_data(self, dir, pars):
            """
            Extract data and manipulations required for the Peters Quinlan model

            Parameters
            ----------
            dir : str
                directory where hierarchical model data (from simulations) is
            pars : dict
                analysis parameters
            """
            obs = {"t":[], "a":[], "e":[], "mass1":[], "mass2":[]}
            i = 0
            for f in dir:
                _logger.info(f"Loading file: {f}")
                hmq = HMQuantitiesBinaryData.load_from_file(f)
                try:
                    idx = hmq.get_idx_in_vec(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
                except ValueError:
                    _logger.warning(f"No data prior to merger! The requested semimajor axis value is {np.nanmedian(hmq.hardening_radius)}, semimajor_axis attribute is: {hmq.semimajor_axis}. This run will not form part of the analysis.")
                    continue
                except AssertionError:
                    _logger.warning(f"Trying to search for value {np.nanmedian(hmq.hardening_radius)}, but an AssertionError was thrown. The array bounds are {min(hmq.semimajor_axis)} - {max(hmq.semimajor_axis)}. This run will not form part of the analysis.")
                    continue
                
                obs["t"].append(hmq.binary_time[idx:])
                # convert to pc
                obs["a"].append(hmq.semimajor_axis[idx:]*1e3)
                obs["e"].append(hmq.eccentricity[idx:])
                obs["mass1"].append([hmq.binary_masses[0]])
                obs["mass2"].append([hmq.binary_masses[1]])
                i += 1
            self.obs = obs
            self.transform_obs("a", "inva", lambda x:1/x)
            self.collapse_observations(["t", "a", "inva", "e"])


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
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        self.plot_generated_quantity_dist(self._latent_qtys, ax=ax, xlabels=self._latent_qtys_labs)
        return ax


    def all_prior_plots(self, figsize=None):
        """
        Prior plots generally required for predictive checks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default None
        """
        fig, ax = plt.subplots(1,2, figsize=figsize, sharex="all")
        for axi in ax: axi.set_xlabel(r"$t/\mathrm{Myr}$")
        ax[0].set_ylabel(r"$a/\mathrm{pc}$")
        ax[1].set_ylabel(r"$e$")
        self.prior_plot(xobs="t", yobs="a", xmodel="t", ymodel="a_prior", ax=ax[0])
        self.prior_plot(xobs="t", yobs="e", xmodel="t", ymodel="e_prior", ax=ax[1])
        # prior latent quantities
        self.plot_latent_distributions(figsize=figsize)




class PQModelSimple(_PQModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.figname_base = f"{self.figname_base}-simple"




class PQModelHierarchy(_PQModelBase):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)
        self.figname_base = f"{self.figname_base}-hierarchy"
        self._hyper_qtys = ["Hps_mean", "Hps_std", "K_mean", "K_std", "ah_mean", "ah_std", "eh_mean", "eh_std"]
        self._hyper_qtys_labs = [r"$\mu_{H\rho/\sigma}$", r"$\sigma_{H\rho/\sigma}$", r"$\mu_{K}$", r"$\sigma_{K}$", r"$\mu_{a_\mathrm{h}}$", r"$\sigma_{a_\mathrm{h}}$", "$\mu_{e_\mathrm{h}}$", r"$\sigma_{e_\mathrm{h}}$"]
        self._labeller_hyper = MapLabeller(dict(zip(self._hyper_qtys, self._hyper_qtys_labs)))
    
    