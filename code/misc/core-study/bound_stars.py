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
extract_data = True

snapdirs = [d.path for d in os.scandir(basedir) if "kick-vel-" in d.name]
snapdirs.sort()

data = {}

if extract_data:
    for snapdir in snapdirs:
        key = f"v{snapdir.split('/')[-1].split('-')[-1]}"
        snapfiles = bgs.utils.get_snapshots_in_dir(os.path.join(snapdir, "output"))
        num_bound_stars = np.zeros_like(snapfiles, dtype=int)
        times = np.full_like(num_bound_stars, np.nan, dtype=float)
        sep = np.full_like(num_bound_stars, np.nan, dtype=float)
        for snap_idx, snapfile in enumerate(snapfiles):
            if snap_idx != 16 and snap_idx != 8: continue
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
    bgs.utils.save_data(data, "bound_snap_8_16.pickle")
else:
    data = bgs.utils.load_data("bound_snap_8_16.pickle")

# plot
fig, ax = plt.subplots(1,1)
ax.set_yscale("log")
ax.set_xlabel("vkick")
ax.set_ylabel("Displacement/kpc")
cmapper, sm = bgs.plotting.create_normed_colours(0, 2000)

idxs_use = (8, 16)

marker_lims = [np.inf, -np.inf]
for k, v in data.items():
    if k == "__githash" or k == "__script":
        continue
    vk = float(k[1:])
    for j, idx_use in enumerate(idxs_use):
        ax.scatter(vk, v["sep"][idx_use], color=bgs.plotting.mplColours()[j], marker="o", s=v["num_bound_stars"][idx_use]/2e2, ec="k", lw=0.5, zorder=2)
        #ax.plot(v["times"], v["sep"], color=cmapper(vk), zorder=0)
        min_s = np.log10(np.min(v["num_bound_stars"])+1)
        max_s = np.log10(np.max(v["num_bound_stars"]))
        if min_s < marker_lims[0]:
            marker_lims[0] = np.floor(min_s)
        if max_s > marker_lims[1]:
            marker_lims[1] = np.ceil(max_s)
# add marker size legend
print(marker_lims)
for msize in np.arange(*marker_lims):
    msize = 10**msize
    ax.scatter([], [], s=msize/2e2, label=int(msize), c="gray")
#ax.set_xlim(0, 0.25)
ax.legend(title="Number bound stars")
plt.colorbar(sm, ax=ax)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "core-study/bound_stars_snap_16.png"))
plt.show()

