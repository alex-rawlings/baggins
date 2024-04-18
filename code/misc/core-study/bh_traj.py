import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


snapfiles = bgs.utils.read_parameters(
        os.path.join(
            bgs.HOME,
            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
        )
    )

snapshots = dict()
for k, v in snapfiles["snap_nums"].items():
    snapshots[k] = os.path.join(
        snapfiles["parent_dir"], f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5"
    )

fig, ax = plt.subplots(1,2, sharex=True)

for snapfile in snapshots.values():
    print(snapfile)
    snap = pygad.Snapshot(snapfile, physical=True)
    ax[0].plot(
        snap.bh["vel"][0,0], snap.bh["vel"][0,1], ls="", marker="o"
    )
    ax[1].plot(
        snap.bh["vel"][0,0], snap.bh["vel"][0,2], ls="", marker="o"
    )
    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()


plt.show()