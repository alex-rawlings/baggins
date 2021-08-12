import pygad.units


__all__ = ['convert_gadget_time']

def convert_gadget_time(snap, new_unit='Gyr'):
    """
    Helper function to convert gadget unit time to physical time
    1 unit time ~ 0.978 Gyr

    Parameters
    ----------
    snap: gadget snapshot object
    new_unit: time units, 'Gyr' or 's'

    Returns
    -------
    float(t): numeric value of time in new_unit
    """
    t = pygad.UnitQty(snap.time, f"({snap.gadget_units['LENGTH']})/({snap.gadget_units['VELOCITY']})").in_units_of(new_unit, subs=snap)
    return float(t)
