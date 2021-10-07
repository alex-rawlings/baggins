import numpy as np
import pygad
from ..general import convert_gadget_time


__all__ = ["snap_num_for_time"]


def snap_num_for_time(snaplist, time_to_find, units="Myr"):
    assert(isinstance(time_to_find, float) or isinstance(time_to_find, int))
    for ind, this_snap in enumerate(snaplist):
        if ind == 0:
            continue
        snap = pygad.Snapshot(this_snap)
        snap.to_physical_units()
        this_time = convert_gadget_time(snap, new_unit=units)
        if time_to_find < this_time:
            idx = ind-1
            break
    else:
        idx = -1
    return idx