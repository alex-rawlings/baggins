import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad
import ketjugw

myr = ketjugw.units.yr * 1e6

def _get_idx_in_vec(t, tarr):
    #get the index of a value t within an array tarr
    #note tarr must have values in ascending order
    if not np.isnan(t):
        if t > tarr.max():
            raise ValueError("Value is larger than the largest array value. Returning index -1")
        elif t < tarr.min():
            raise ValueError("Value is smaller than the smallest array value. Returning index 0")
        else:
            return np.argmax(t < tarr)
    else:
        raise ValueError("t must not be nan")


snapdir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/000/output"

bhfile = os.path.join(snapdir, "ketju_bhs_cp.hdf5")
snaplist = cmf.utils.get_snapshots_in_dir(snapdir)

bh1, bh2, merged = cmf.analysis.get_bound_binary(bhfile)
orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)

t = np.full_like(snaplist, np.nan, dtype=float)
J_lc = np.full_like(t, np.nan)
num_J_lc = np.full_like(t, np.nan)
theta = np.full_like(t, np.nan)

for i, snapfile in enumerate(snaplist):
    if i>8: break
    print(snapfile)
    snap = pygad.Snapshot(snaplist[i], physical=True)
    # as we are interested in flow rates about binary, set binary as
    # the centre of mass
    this_centre_pos = pygad.analysis.center_of_mass(snap.bh)
    this_centre_vel = pygad.analysis.mass_weighted_mean(snap.bh, "vel")
    snap["pos"] -= this_centre_pos
    snap["vel"] -= this_centre_vel
    t[i] = cmf.general.convert_gadget_time(snap, new_unit="Myr")
    try:
        idx = _get_idx_in_vec(t[i], orbit_params["t"]/myr)
        # determine loss cone J, _a is semimajor axis from ketjugw as 
        # pygad scalar object
        _a = pygad.UnitScalar(orbit_params["a_R"][idx]/ketjugw.units.pc, "pc")
        J_lc[i] = cmf.analysis.loss_cone_angular_momentum(snap, _a)
        # determine the magnitude of the angular momentum
        star_J_mag = pygad.utils.geo.dist(snap.stars["angmom"])
        print(star_J_mag.min())
        num_J_lc[i] = np.sum(star_J_mag < J_lc[i])
    except ValueError:
        J_lc[i] = np.nan
        num_J_lc[i] = np.nan
    theta[i] = cmf.analysis.angular_momentum_difference_gal_BH(snap)
    snap.delete_blocks()

plt.plot(t, theta, "-o")
plt.show()