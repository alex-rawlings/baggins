import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use
    use("Agg")
    import matplotlib.pyplot as plt
import baggins as bgs
import pygad
from datetime import datetime

bgs.plotting.check_backend()

basedir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
extract_data = False

snapdirs = [d.path for d in os.scandir(basedir) if "kick-vel-" in d.name and (float(d.name[-4:]) < 1080 or float(d.name[-4:]) >1990)]
snapdirs.sort()

data = {}

if extract_data:
    for snapdir in snapdirs:
        print(f"In {snapdir}")
        key = f"v{snapdir.split('/')[-1].split('-')[-1]}"
        snapfiles = bgs.utils.get_snapshots_in_dir(os.path.join(snapdir, "output"))[:4]
        num_bound_stars = np.zeros_like(snapfiles, dtype=int)
        times = np.full_like(num_bound_stars, np.nan, dtype=float)
        sep = np.full_like(num_bound_stars, np.nan, dtype=float)
        for snap_idx, snapfile in enumerate(snapfiles):
            snap = pygad.Snapshot(snapfile, physical=True)
            if len(snap.bh) > 1:
                print(f"There are {len(snap.bh)} BHs in snapshot {snapfile}")
            else:
                times[snap_idx] = bgs.general.convert_gadget_time(snap)
                sep[snap_idx] = bgs.mathematics.radial_separation(snap.bh["pos"])[0]
                now = datetime.now()
                num_bound_stars[snap_idx] = len(bgs.analysis.find_individual_bound_particles(snap))
                print(f"{snap_idx}/{len(num_bound_stars)}: Bound stars ({num_bound_stars[snap_idx]}) found in {datetime.now()-now}")
            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap
        print(f"Complete for {snapdir}")
        data[key] = dict(
            times = times,
            sep = sep,
            num_bound_stars = num_bound_stars
        )
    bgs.utils.save_data(data, "bound_snap_up_to_4.pickle")
else:
    data = bgs.utils.load_data("bound_snap_up_to_4.pickle")

# PARAMETERS TO EDIT
# TODO: should read these from snapshot and save
stellar_mass = 5e4
binary_mass = 2 * 2.93e9
total_stellar_mass = 1.38e11

# plot
fig, ax = plt.subplots(1,1)
ax.set_yscale("log")
ax.set_xlabel("vkick")
ax.set_ylabel("Displacement/kpc")
cmapper, sm = bgs.plotting.create_normed_colours(0, np.nanmax([np.nanmax(v["times"]) for v in data.values()]))

idxs_use = (0,1,2,3,4,)

marker_lims = [np.inf, -np.inf]
for k, v in data.items():
    if k == "__githash" or k == "__script":
        continue
    vk = float(k[1:])
    still_binary = True
    for j, idx_use in enumerate(idxs_use):
        try:
            ax.scatter(vk, v["sep"][idx_use], marker="o", s=v["num_bound_stars"][idx_use]/2e2, ec="k", lw=0.5, zorder=2, color=cmapper(v["times"][idx_use]))
            #ax.plot(v["times"], v["sep"], color=cmapper(vk), zorder=0)
            min_s = np.log10(np.min(v["num_bound_stars"])+1)
            max_s = np.log10(np.max(v["num_bound_stars"]))
            if min_s < marker_lims[0]:
                marker_lims[0] = np.floor(min_s)
            if max_s > marker_lims[1]:
                marker_lims[1] = np.ceil(max_s)
            if still_binary and v["num_bound_stars"][idx_use] > 0:
                print(f"{vk:04.0f}: {v['num_bound_stars'][idx_use]*stellar_mass/binary_mass:.2e} MBH - {v['num_bound_stars'][idx_use]*stellar_mass:.2e} Msol - {v['num_bound_stars'][idx_use]*stellar_mass/total_stellar_mass:.2e} MStar")
                still_binary = False
        except IndexError:
            continue
# add marker size legend
print(marker_lims)
for msize in np.arange(*marker_lims):
    msize = 10**msize
    ax.scatter([], [], s=msize/2e2, label=int(msize), c="gray")
#ax.set_xlim(0, 0.25)
ax.legend(title="Number bound stars")
plt.colorbar(sm, ax=ax)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "core-study/bound_stars_snap_16.png"))
#plt.show()

