import os.path
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Investigate triaxiality",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
)
args = parser.parse_args()


snapfiles = bgs.utils.get_snapshots_in_dir(f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output")[:60]

times = np.full_like(snapfiles, np.nan, dtype=float)
ba = np.full_like(times, np.nan)
ca = np.full_like(times, np.nan)
major_axis = np.full((len(ba), 3), np.nan)
half_mass_rad = np.full_like(ba, np.nan)
dm_within_re = np.full_like(ba, np.nan)

for i, snapfile in tqdm(enumerate(snapfiles), total=len(snapfiles), desc="Analysing snapshots"):
    #print(f"Doing snapshot {bgs.general.get_snapshot_number(snapfile)}")
    snap = pygad.Snapshot(snapfile, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)
    times[i] = bgs.general.convert_gadget_time(snap)
    half_mass_rad[i] = pygad.analysis.half_mass_radius(snap.stars[pygad.BallMask(30)])
    dm_within_re[i] = bgs.analysis.inner_DM_fraction(snap, half_mass_rad[i])
    half_mass_rad_mask = pygad.BallMask(half_mass_rad[i])
    (ba[i], ca[i]), eigvec = bgs.analysis.get_galaxy_axis_ratios(snap, bin_mask=half_mass_rad_mask, return_eigenvectors=True)
    major_axis[i,:] = eigvec[:,0]

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

fig = plt.figure()
ax = np.array([fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)])
fig.suptitle(f"{args.kv} km/s")

ax[0].plot(times, ba, label="b/a")
ax[0].plot(times, ca, label="c/a")
ax[0].legend()
ax[0].set_xlabel("t/Gyr")
ax[0].set_ylabel("Axis ratios")

major_axis_S = bgs.mathematics.convert_cartesian_to_spherical(major_axis)
ax[1].plot(times, major_axis_S[:,1] * 180/np.pi, label="theta")
ax[1].plot(times, major_axis_S[:,2] * 180/np.pi, label="phi")
ax[1].legend()
ax[1].set_xlabel("t/Gyr")
ax[1].set_ylabel("Major axis direction")


'''
cmapper, sm = bgs.plotting.create_normed_colours(min(times), max(times))
out_of_page = major_axis_S[:,1] > 0
ax[1].scatter(major_axis_S[out_of_page, 2], major_axis_S[out_of_page,0], marker="o", color=cmapper(times[out_of_page]))
ax[1].scatter(major_axis_S[~out_of_page, 2], major_axis_S[~out_of_page,0], marker="x", color=cmapper(times[~out_of_page]))
ax[1].set_rticks([0.5, 1])
ax[1].set_rmax(1.5)
ax[1].grid(True)'''

ax[2].plot(times, half_mass_rad)
ax[2].set_xlabel("t/Gyr")
ax[2].set_ylabel("R_half/kpc")

ax[3].plot(times, dm_within_re)
ax[3].set_xlabel("t/Gyr")
ax[3].set_ylabel("Frac. DM < R_half")

bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/triax.png"))