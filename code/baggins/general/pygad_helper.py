import numpy as np
import pygad.units
from ..env_config import _cmlogger

__all__ = ["convert_gadget_time", "unit_as_str", "particle_ages"]


_logger = _cmlogger.getChild(__name__)


def convert_gadget_time(snap, new_unit="Gyr"):
    """
    Helper function to convert gadget unit time to physical time
    1 unit time ~ 0.978 Gyr

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to convert time for
    new_unit : str, optional
        new time units, by default "Gyr"

    Returns
    -------
    : pygad.UnitArr
        time in new_unit
    """
    t = pygad.UnitQty(
        snap.time, f"({snap.gadget_units['LENGTH']})/({snap.gadget_units['VELOCITY']})"
    ).in_units_of(new_unit, subs=snap)
    return float(t)


def unit_as_str(u):
    """
    Return the units of a pygad UnitArr as a string

    Parameters
    ----------
    u : pygad.Units.unit
        _description_

    Returns
    -------
    : str
        string representation
    """
    return str(u).strip("[]")


def particle_ages(subsnap, unit="Gyr"):
    """
    Determine the age of particles in a snapshot, as the pygad 'age' block
    doesn't seem to work.

    Parameters
    ----------
    subsnap : pygad.SubSnapshot
        subsnapshot of particle family to determine age of, e.g. snap.stars
    unit : str, optional
        units for age, by default "Gyr"

    Returns
    -------
    : pygad.UnitArr
        age of particles
    """
    try:
        assert "form_time" in subsnap.available_blocks()
    except AssertionError:
        _logger.exception("subsnapshot must have block 'form_time'!", exc_info=True)
        raise
    t = pygad.UnitArr(
        (subsnap.root.time - subsnap["form_time"].view(np.ndarray)),
        f"({subsnap.gadget_units['LENGTH']})/({subsnap.gadget_units['VELOCITY']})",
    ).in_units_of(unit, subs=subsnap.root)
    return t
