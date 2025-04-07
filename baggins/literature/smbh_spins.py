import os.path
import numpy as np
import scipy.stats
import scipy.integrate
from baggins.mathematics import (
    uniform_sample_sphere,
    convert_spherical_to_cartesian,
    radial_separation,
    create_orthonormal_basis_from_vec,
)

__all__ = [
    "zlochower_dry_spins",
    "zlochower_hot_spins",
    "zlochower_cold_spins",
    "SMBHSpins",
]

# dry accretion model
zlochower_dry_spins = dict(spin_mag_a=10.5868, spin_mag_b=4.66884)

# wet accretion, hot model
zlochower_hot_spins = dict(spin_mag_a=3.212, spin_mag_b=1.563)

# wet accretion, cold model
zlochower_cold_spins = dict(spin_mag_a=5.935, spin_mag_b=1.856)


class SMBHSpins:
    def __init__(self, magnitude, direction="uniform", rng=None):
        """
        Class to generate SMBH spins

        Parameters
        ----------
        magnitude : str
            model for spin magnitude
        direction : str, optional
            model for spin direction, by default "uniform"
        rng : np.random.Generator, optional
        random number generator, by default None (creates a new instance)
        """
        assert magnitude in ("zlochower_hot", "zlochower_cold", "zlochower_dry")
        assert direction in ("uniform", "skewed")
        if magnitude == "zlochower_hot":
            self.alpha_params = zlochower_hot_spins
        elif magnitude == "zlochower_cold":
            self.alpha_params = zlochower_cold_spins
        else:
            self.alpha_params = zlochower_dry_spins
        if direction == "uniform":
            self.theta_func = self._uniform_dir_on_sphere
        else:
            # create the CDF from Lousto+10 data
            d = np.loadtxt(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "literature_data/lousto_10.txt",
                ),
                delimiter=",",
                skiprows=1,
            )
            cdf = scipy.integrate.cumulative_trapezoid(
                d[:, 1], d[:, 0], initial=0
            ) / scipy.integrate.trapezoid(d[:, 1], d[:, 0])
            self.theta_func = lambda n, L: self._skewed_dir_on_sphere(
                n, L, lambda x: np.interp(x, cdf, d[:, 0])
            )
        if rng is None:
            self._rng = np.random.default_rng()

    def _uniform_dir_on_sphere(self, n, **kwargs):
        """
        Generate unit vectors uniformly on the sphere

        Parameters
        ----------
        n : int
            number of draws

        Returns
        -------
        : np.ndarray
            unit vectors
        """
        rtp = np.ones((n, 3))
        rtp[:, 1], rtp[:, 2] = uniform_sample_sphere(n=n, rng=self._rng)
        return convert_spherical_to_cartesian(rtp)

    def _skewed_dir_on_sphere(self, n, L, f):
        """
        Generate unit vectors biased towards binary angular momentum following
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.084023

        Parameters
        ----------
        n : int
            number of draws
        L : array-like
            binary angular momentum vector
        f : callable
            function to generate offset angle

        Returns
        -------
        : np.ndarray
            unit vectors
        """
        u = self._rng.uniform(0, 1, size=n)
        e1, e2, e3 = create_orthonormal_basis_from_vec(L)

        # Generate random angles
        phi = np.random.uniform(0, 2 * np.pi, n)  # Azimuthal angle
        psi = f(u)  # Offset angle

        # Convert to Cartesian coordinates in local frame
        return (
            np.outer(np.cos(psi), e1)
            + np.outer(np.sin(psi) * np.cos(phi), e2)
            + np.outer(np.sin(psi) * np.sin(phi), e3)
        )

    def sample_spin_magnitudes(self, n):
        """
        Generate spin magnitude from assumed model.

        Parameters
        ----------
        n : int
            number of draws

        Returns
        -------
        : np.ndarray
            spin magnitudes
        """
        return scipy.stats.beta.rvs(
            *self.alpha_params.values(),
            random_state=self._rng,
            size=n,
        )

    def sample_spin_directions(self, n, **kwargs):
        """
        Public method to generate spin directions

        Parameters
        ----------
        n : int
            number of draws

        Returns
        -------
        : np.ndarray
            unit vectors
        """
        return self.theta_func(n, **kwargs)

    def sample_spins(self, n, **kwargs):
        """
        Generate spins (magnitudes and directions sampled).

        Parameters
        ----------
        n : int
            number of draws

        Returns
        -------
        : np.ndarray
            SMBH spin vectors
        """
        spindir = self.sample_spin_directions(n, **kwargs)
        alpha = np.atleast_2d(self.sample_spin_magnitudes(n)).T
        spins = alpha * spindir
        assert np.all(radial_separation(spins) <= 1.0)
        return spins
