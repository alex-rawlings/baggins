import os
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad
import dask
import multiprocessing as mp


basedir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"

snapdirs = [d.path for d in os.scandir(basedir) if "kick-vel-0" in d.name and "0990" not in d.name]
snapdirs.sort()

fig, ax = plt.subplots(1,1)
ax.set_xlabel("t/Gyr")
ax.set_ylabel("Bound stars")

if __name__ == "__main__":
    for snapdir in snapdirs:
        #print(f"In directory: {snapdir}")
        if "0900" not in snapdir: continue
        snapfiles = cmf.utils.get_snapshots_in_dir(os.path.join(snapdir, "output"))
        num_bound_stars = np.zeros_like(snapfiles, dtype=int)
        times = np.full_like(num_bound_stars, np.nan, dtype=float)
        sep = np.full_like(num_bound_stars, np.nan, dtype=float)
        for snap_idx, snapfile in enumerate(snapfiles):
            #if snap_idx != 0 and snap_idx != len(snapfiles)-1: continue
            #print(f"{snap_idx / (len(snapfiles)-1)*100:.1f}% complete                  ", end="\r")
            snap = pygad.Snapshot(snapfile, physical=True)
            if len(snap.bh) > 1:
                print(f"There are {len(snap.bh)} BHs in snapshot {snapfile}")
            else:
                times[snap_idx] = cmf.general.convert_gadget_time(snap)
                sep[snap_idx] = cmf.mathematics.radial_separation(snap.bh["pos"])[0]
                num_bound_stars[snap_idx] = len(cmf.analysis.find_bound_particles(snap))
            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap
            #print("----")
        ax.semilogy(times, num_bound_stars)
        print(f"Complete for {snapdir}")
        plt.show()
        quit()


