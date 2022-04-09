import pygad.units


__all__ = ["convert_gadget_time", "unit_as_str"]


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
    t = pygad.UnitQty(snap.time, f"({snap.gadget_units['LENGTH']})/({snap.gadget_units['VELOCITY']})").in_units_of(new_unit, subs=snap)
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
