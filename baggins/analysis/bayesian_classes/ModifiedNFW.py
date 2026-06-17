import os.path
from baggins.env_config import _cmlogger, baggins_dir
from baggins.analysis.bayesian_classes.ABGDensityModels import ABGDensityModelSimple

__all__ = ["ModifiedNFWModelSimple"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/modified-NFW/{f.rstrip('.stan')}.stan")


class ModifiedNFWModelSimple(ABGDensityModelSimple):
    def __init__(self, figname_base, rng=None):
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
        self._latent_qtys_labs.append(r"$\gamma$")
        self._latent_qtys_posterior_labs.append(r"$\gamma$")
        self._make_latent_labellers()
        _logger.debug(f"Latent quantities are {self.latent_qtys}")

    def extract_data(self, snapfile=None, extent=300, bin_count=2e5, family="dm"):
        super().extract_data(snapfile, extent, bin_count, family)
        self.figname_base = os.path.join(
            super().figname_base, f"{self.merger_id}/{self.merger_id}-simple"
        )

    def add_guiding_NFW(self, ax, rS, g, N=5, offset=0.5, **kwargs):
        """
        Plot NFW profile to guide eye. See add_guiding_profiles() for details.
        """
        kwargs.setdefault("label", "mNFW")
        self.add_guiding_profiles(
            ax=ax, a=1, b=3, g=g, rS=rS, N=N, offset=offset, **kwargs
        )

    def add_guiding_Plummer(self, **kwargs):
        raise RuntimeError(
            "Guiding Plummer profile not available for 'ModifiedNFWModelSimple'"
        )
