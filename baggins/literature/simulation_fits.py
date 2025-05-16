import numpy as np
import scipy.integrate

__all__ = ["Fakhouri2010_merger_rate", "Fakhouri2010_cumulative_mergers"]


def Fakhouri2010_merger_rate(M, mass_ratio, z):
    """
    Determine the merger rate from Fakhouri et al. 2010.
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.406.2267F/abstract

    Parameters
    ----------
    M : float
        halo virial mass
    mass_ratio : float
        merger mass ratio
    z : float
        redshift

    Returns
    -------
    : float
        merger rate
    """
    A = 0.0104
    xi_tilde = 9.72e-3
    alpha = 0.133
    beta = -1.995
    gamma = 0.263
    eta = 0.0993
    return (
        A
        * (M / 1e12) ** alpha
        * mass_ratio**beta
        * np.exp((mass_ratio / xi_tilde) ** gamma)
        * (1 + z) ** eta
    )


def Fakhouri2010_cumulative_mergers(M, min_mass_ratio, redshift=1, redshift0=0):
    """
    Cumulative merger rate given by Fakhouri et al. 2010.
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.406.2267F/abstract

    Parameters
    ----------
    M : float
        halo virial mass
    min_mass_ratio : float
        minimum merger mass ratio
    redshift : float, optional
        upper bound on redshift, by default 1
    redshift0 : float, optional
        lower bound on redshift, by default 0

    Returns
    -------
    : float
        cumulative number of mergers of a halo in the given redshift interval
    """
    return scipy.integrate.dblquad(
        lambda xi, z: Fakhouri2010_merger_rate(M=M, mass_ratio=xi, z=z),
        redshift0,
        redshift,
        min_mass_ratio,
        1,
    )[0]
