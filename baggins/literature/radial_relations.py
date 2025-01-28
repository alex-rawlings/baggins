"""
A collection of miscellaneous relations that are somehow dependent on a radial
value.
"""

__all__ = ["OsipkovMerritt", "Sahu20", "Merritt09"]


def OsipkovMerritt(r, ra):
    """
    Define the theoretical anisotropy value of an Osipkov-Merritt model with
    parameter ra

    Parameters
    ----------
    r : np.ndarray
        radial values
    ra : float
        anisotropy radius

    Returns
    -------
    : np.ndarray
        anisotropy profile beta
    """
    return 1 / (1 + (ra / r) ** 2)


def Sahu20(logRe):
    """
    Define the Sahu+20 spheroidal mass - Re relation
    https://ui.adsabs.harvard.edu/abs/2020ApJ...903...97S/abstract

    Parameters
    ----------
    logRe : np.ndarray
        logarithm of effective radius [log(R/kpc)]

    Returns
    -------
    : np.ndarray
        expected logarithm of spheroidal mass [log(M/Msol)]
    """
    return 1.08 * logRe + 10.32


def Merritt09(bh_mass):
    """
    Define the Merritt+09 influence radius - bh mass relation.
    https://ui.adsabs.harvard.edu/abs/2009ApJ...699.1690M/abstract

    Parameters
    ----------
    bh_mass : float
        BH mass [Msol]

    Returns
    -------
    : float
        influence radius [kpc]
    """
    return 35e-3 * (bh_mass / 1e8) ** 0.56
