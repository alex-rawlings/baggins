import numpy as np
import pygad.units
from env_config import _cmlogger

__all__ = ["convert_gadget_time", "unit_as_str", "particle_ages", "snap_num_for_time"]


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


def snap_num_for_time(snaplist, time_to_find, units="Myr", method="floor"):
    """
    Determine the snapshot number for the given time. May result in the
    last snapshot in the list to be returned if the given time is much later
    than the snapshots.

    Parameters
    ----------
    snaplist : list
        snapshot files
    time_to_find : float, int, pygad.UnitArr
        time we want to find
    units : str, optional
        units of the time, by default "Myr"
    method : str, optional
        one of
        - 'floor': last snapshot before the given time
        - 'ceil': first snapshot after the given time
        - 'nearest': snapshot closest to the given time,
        by default "floor"

    Returns
    -------
    idx: int
        index in the list of snapshots corresponding to the desired time by the
        desired method

    Raises
    ------
    ValueError
        if given method is invalid
    """
    if method not in ["floor", "ceil", "nearest"]:
        raise ValueError("method must be one of 'floor', 'ceil', or 'nearest'.")
    assert isinstance(time_to_find, (float, int, pygad.UnitArr))
    for ind, this_snap in enumerate(snaplist):
        # TODO more efficient to read directly from hdf5 instead of loading snapshot?
        snap = pygad.Snapshot(this_snap, physical=True)
        this_time = convert_gadget_time(snap, new_unit=units)
        snap.delete_blocks()
        pygad.gc_full_collect()
        del snap
        if ind == 0:
            prev_time = this_time
            continue
        if time_to_find < this_time:
            if method == "floor":
                idx = ind - 1
            elif method == "ceil":
                idx = ind
            else:
                if this_time - time_to_find > time_to_find - prev_time:
                    idx = ind - 1  # closest snap is the one before
                else:
                    idx = ind  # closest snap is the one after
            break
        prev_time = this_time
    else:
        idx = len(snaplist) - 1
        _logger.warning("Returning the final snapshot in the list!")
    return idx
