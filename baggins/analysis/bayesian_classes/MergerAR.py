import os.path
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import arviz as az
from baggins.analysis.bayesian_classes.StanModel import HierarchicalModel_2D
from baggins.analysis.analyse_ketju import get_bound_binary
from baggins.env_config import baggins_dir, _cmlogger
from baggins.general.units import Gyr
from baggins.utils import get_ketjubhs_in_dir


__all__ = ["MergerAutoRegression"]

_logger = _cmlogger.getChild(__name__)


def get_stan_file(f):
    return os.path.join(baggins_dir, f"stan/autoregression/{f.rstrip('.stan')}.stan")


class MergerAutoRegression(HierarchicalModel_2D):
    def __init__(self, figname_base, rng=None, thin=100):
        super().__init__(
            model_file=get_stan_file("ar1"),
            prior_file=get_stan_file("ar1"),  # TODO update this!
            figname_base=figname_base,
            rng=rng,
        )
        self.thin = thin
        self._latent_qtys = ["a1", "a2", "sigma"]
        self._latent_qtys_labs = [r"$a_1$", r"$a_2$", r"$\sigma$"]
        self._labeller_latent = az.labels.MapLabeller(
            dict(zip(self._latent_qtys, self._latent_qtys_labs))
        )
        self._folded_qtys = ["y"]
        self._folded_qtys_labs = [r"$\Delta e$"]
        self._folded_qtys_posterior = [f"{v}_posterior" for v in self._folded_qtys]

    @property
    def folded_qtys(self):
        return self._folded_qtys

    @property
    def folded_qtys_posterior(self):
        return self._folded_qtys_posterior

    @property
    def latent_qtys(self):
        return self._latent_qtys

    @property
    def latent_qtys_posterior(self):
        return self._latent_qtys_posterior

    def extract_data(self, d):
        d = self._get_data_dir(d)
        d = get_ketjubhs_in_dir(d)
        obs = {"t": [], "ecc": [], "a": [], "ecc_diff": []}
        for kf in d:
            _logger.info(f"Reading ketju data: {kf}")
            bh1, bh2, *_ = get_bound_binary(kf)
            op = ketjugw.orbital_parameters(bh1, bh2)
            obs["t"].append(bh1.t / Gyr)
            obs["a"].append(op["a_R"] / ketjugw.units.pc)
            obs["ecc"].append(op["e_t"])
            _ecc_diff = np.zeros_like(obs["ecc"][-1])
            _ecc_diff[1:] = np.diff(obs["ecc"][-1])
            obs["ecc_diff"].append(_ecc_diff)
        self.obs = obs
        self.collapse_observations(["t", "a", "ecc", "ecc_diff"])
        self.thin_observations(self.thin)

    def set_stan_data(self):
        super().set_stan_data()
        self.stan_data = dict(
            N=self.num_obs_collapsed - 1,
            t=self.obs_collapsed["t"][1:],
            y=self.obs_collapsed["ecc_diff"][1:],
        )

    def _set_stan_data_OOS(self):
        return super()._set_stan_data_OOS()

    def sample_model(self, sample_kwargs={}, diagnose=True):
        """
        Wrapper around StanModel.sample_model() to handle determining num_OOS
        from previous sample.
        """
        super().sample_model(sample_kwargs=sample_kwargs, diagnose=diagnose)
        if self._loaded_from_file:
            raise NotImplementedError
            self._determine_num_OOS(self._folded_qtys_posterior[0])
            self._set_stan_data_OOS()

    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        v = super().sample_generated_quantity(gq, force_resample, state)
        if gq in self.folded_qtys or gq in self.folded_qtys_posterior:
            idxs = self._get_GQ_indices(state)[:-1]
            return v[..., idxs]
        else:
            return v

    def all_posterior_pred_plots(self):
        self.parameter_diagnostic_plots(
            self._latent_qtys, labeller=self._labeller_latent
        )

        # posterior predictive check
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(r"$t/\mathrm{Gyr}$")
        ax.set_ylabel(self._folded_qtys_labs[0])
        self.plot_predictive(
            xmodel="t",
            ymodel=f"{self._folded_qtys_posterior[0]}",
            xobs="t",
            yobs="ecc_diff",
            ax=ax,
        )
        plt.close()

        _ecc_diff = self.sample_generated_quantity("y_posterior")
        _ecc_diff[0] = self.obs_collapsed["ecc"][0]
        ecc = np.cumsum(_ecc_diff, axis=0)

        # posterior predictive check
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$t/\mathrm{Gyr}$")
        ax.set_ylabel(r"$e$")
        self.plot_predictive(
            xmodel="t",
            ymodel=ecc,
            xobs="t",
            yobs="ecc",
            ax=ax,
        )

        return ax
