import os.path
import numpy as np
from baggins.env_config import _cmlogger, baggins_dir
from baggins.analysis.bayesian_classes.ABGDensityModels import ABGDensityModelSimple

__all__ = ["ModifiedNFWModelSimple", "ModifiedNFWSpikeModelSimple"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/modified-NFW/{f.rstrip('.stan')}.stan")


class ModifiedNFWModelSimple(ABGDensityModelSimple):
    def __init__(self, figname_base, rng=None):
        """
        Model for non-hierarchical modified-NFW profile. It inherits as a special case of the more general Alpha-Beta-Gamma profile.

        Parameters
        ----------
        figname_base : str
            base string for figure names
        rng : np.random.Generator, optional
            random number generator, by default None
        """
        super().__init__(figname_base, rng)
        # set different model files
        self._model_file = get_stan_file("mNFW_simple")
        self._prior_file = get_stan_file("mNFW_simple_prior")
        # update latent parameters
        _logger.debug(f"Latent quantities are initially {self.latent_qtys}")
        for p in ["log10a", "b", "g"]:
            i = self._latent_qtys.index(p)
            self._latent_qtys.pop(i)
            self._latent_qtys_posterior.pop(i)
            self._latent_qtys_labs.pop(i)
            self._latent_qtys_posterior_labs.pop(i)
        self._latent_qtys.append("log10g")
        self._latent_qtys_posterior.append("g")
        self._latent_qtys_labs.append(r"$\log_{10}\gamma$")
        self._latent_qtys_posterior_labs.append(r"$\gamma$")
        self._make_latent_labellers()
        _logger.debug(f"Latent quantities are {self.latent_qtys}")

    def extract_data(self, snapfile=None, **kwargs):
        """
        Extract data to perform inference on.

        Parameters
        ----------
        snapfile : str, optional
            name of snapshot, by default None

        Returns
        -------
        snap : pygad.Snapshot
            snapshot used for fitting
        """
        kwargs.setdefault("extent", 300)
        kwargs.setdefault("n_start", 500)
        kwargs.setdefault("n_end", int(5e4))
        kwargs["family"] = "dm"
        snap = super().extract_data(snapfile, **kwargs)
        self.figname_base = os.path.join(
            super().figname_base, f"{self.merger_id}/{self.merger_id}-simple"
        )
        return snap

    def add_guiding_NFW(self, ax, rS, g, N=5, offset=0.5, **kwargs):
        """
        Plot NFW profile to guide eye. See add_guiding_profiles() for details.
        """
        kwargs.setdefault("label", "mNFW")
        self.add_guiding_profiles(
            ax=ax, a=1, b=3, g=g, rS=rS, N=N, offset=offset, **kwargs
        )

    def add_guiding_Plummer(self, **kwargs):
        """
        Override method to prevent Plummer profiles from being added to the plot.

        Raises
        ------
        RuntimeError
            if called
        """
        raise RuntimeError(
            "Guiding Plummer profile not available for 'ModifiedNFWModelSimple'"
        )


class ModifiedNFWSpikeModelSimple(ModifiedNFWModelSimple):
    def __init__(self, figname_base, rng=None):
        """
        Model for a non-hierarchical modified-NFW profile with a central DM
        spike carved out by the growth of the central black hole, following
        Alonso-Alvarez, Cline & Dewar (2024), arXiv:2401.14450.

        Parameters
        ----------
        figname_base : str
            base string for figure names
        rng : np.random.Generator, optional
            random number generator, by default None
        """
        super().__init__(figname_base, rng)
        # set different model files
        self._model_file = get_stan_file("mNFWspike_simple")
        self._prior_file = get_stan_file("mNFWspike_simple_prior")
        self._M_BH = None
        # update latent parameters
        self._latent_qtys.append("gamma_sp")
        self._latent_qtys_posterior.append("gamma_sp")
        self._latent_qtys_labs.append(r"$\gamma_\mathrm{sp}$")
        self._latent_qtys_posterior_labs.append(r"$\gamma_\mathrm{sp}$")
        self._make_latent_labellers()

    @property
    def M_BH(self):
        return self._M_BH

    @M_BH.setter
    def M_BH(self, m):
        self._M_BH = m

    def extract_data(self, snapfile=None, **kwargs):
        """
        Extract data to perform inference on, additionally recording the
        total mass of the central black hole(s) that seeded the DM spike.
        See docs for ModifiedNFWModelSimple.extract_data().

        Parameters
        ----------
        snapfile : str, optional
            name of snapshot, by default None

        Returns
        -------
        snap : pygad.Snapshot
            snapshot used for fitting
        """
        snap = super().extract_data(snapfile, **kwargs)
        self.M_BH = float(np.sum(snap.bh["mass"]))
        _logger.debug(f"Black hole mass set to {self.M_BH:.3e} Msun")
        return snap

    def set_stan_data(self, **kwargs):
        """See docs for ABGDensityModelSimple.set_stan_data()"""
        try:
            assert self.M_BH is not None
        except AssertionError:
            _logger.exception(
                "'M_BH' has not been set! Either call extract_data() on a "
                "snapshot, or set the 'M_BH' property directly.",
                exc_info=True,
            )
            raise
        super().set_stan_data(**kwargs)
        self.stan_data = {"M_BH": self.M_BH}
