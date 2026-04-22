from baggins.env_config import _cmlogger

__all__ = ["e_from_rperi", "ensure_reasonable_particle_counts"]

_logger = _cmlogger.getChild(__name__)


def e_from_rperi(x, a=0.320, b=1.629, c=0.176):
    """
    Determine eccentricity from r/Rvir using fit to Khochfar & Burkett 2006
    Fig. 6

    Parameters
    ----------
    x : np.ndarray
        normalised rperi values (normalised to the virial radius of the larger
        progenitor)
    a : float, optional
        shape parameter, by default 0.320
    b : float, optional
        shape parameter, by default 1.629
    c : float, optional
        shape parameter, by default 0.176

    Returns
    -------
    : np.ndarray
        eccentricity of approach
    """
    return (1 + (x / a) ** b) ** (-c)


def ensure_reasonable_particle_counts(gal, threshold=1e4):
    """
    Protect against user-error where due to e.g. mass conversion no or few particles may be generated.

    Parameters
    ----------
    gal : merger_ic_generator.System
        initialised galaxy
    threshold : int, float, optional
        threshold for particle count below which to raise warning, by default 1e4
    """
    threshold = int(threshold)
    for ptype, count in gal.particle_counts.items():
        if ptype.name.lower() != "bh" and count < threshold:
            _logger.warning(
                f"Particle type {ptype.name} has only {count} generated particles!"
            )
