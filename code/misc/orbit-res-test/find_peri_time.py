import numpy as np
import scipy.spatial.distance, scipy.signal
import matplotlib.pyplot as plt
import ketjugw
import pygad
import cm_functions as cmf

data_path = "/scratch/pjohanss/arawling/collisionless_merger/softening-test/0-04/output/"
bh_file = "ketju_bhs_cp.hdf5"
bhs = ketjugw.data_input.load_hdf5(data_path+bh_file)
bh1, bh2 = bhs.values()

orbit_energy = ketjugw.orbital_energy(bh1, bh2)

myr = 1e6 * ketjugw.units.yr
kpc = 1e3 * ketjugw.units.pc

peritime, idx, radial_sep = cmf.analysis.find_pericentre_time(bh1, bh2, return_sep=True)

snap_files = cmf.utils.get_snapshots_in_dir(data_path)
times = np.full_like(snap_files, np.nan, dtype=float)
bhseps = np.full_like(snap_files, np.nan, dtype=float)
for i, snapfile in enumerate(snap_files):
    snap = pygad.Snapshot(snapfile)
    snap.to_physical_units()
    times[i] = cmf.general.convert_gadget_time(snap)*1e3
    bhseps[i] = np.linalg.norm(snap.bh["pos"][0,:] - snap.bh["pos"][1,:])

snapNum = cmf.analysis.snap_num_for_time(snap_files, peritime[0])

fig, (ax, ax1) = plt.subplots(2,1,sharex="all")
ax.plot(bh1.t/myr, radial_sep, label="ketju_bhs")
ax.scatter(peritime, radial_sep[idx], c="tab:orange", marker=".", zorder=10, label="pericentre")
ax.scatter(times[snapNum], bhseps[snapNum], c="tab:purple", s=200, marker="*", label="extraction")
ax.scatter(times, bhseps, c="tab:green", label="snaps")
cmf.plotting.shade_bool_regions(ax, bh1.t/myr, orbit_energy<0, alpha=0.4, color="tab:red")
ax1.set_xlabel('t/Myr')
ax.set_ylabel("BH Separation [kpc]")
ax.legend()
ax1.plot(bh1.t/myr, orbit_energy)
ax1.set_ylabel("Orbital Energy")
plt.tight_layout()
plt.savefig("/users/arawling/figures/soft-test/orbit.png", dpi=300)
plt.show()
