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

fig, ax = plt.subplots(1,3, sharex=True)

cmapper, sm = bgs.plotting.create_normed_colours(0, 2000)
plot_kwargs = {"ls":"", "marker":"o", "mec":"k", "mew":0.5}

for k, v in snapfiles["snap_nums"].items():
    vkick = float(k.lstrip("v"))
    snapfile = os.path.join(
        snapfiles["parent_dir"], f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5"
    )
    print(snapfile)
    snap = pygad.Snapshot(snapfile, physical=True)
    for i in range(3):
        ax[i].plot(
            vkick, snap.bh["pos"][0,i], c=cmapper(vkick), **plot_kwargs
        )
    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

'''for axi in ax:
    axi.set_aspect("equal")'''


plt.show()