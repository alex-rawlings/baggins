import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad



snapdir = "/scratch/pjohanss/arawling/gadget4-ketju/A-C-3.0-0.05/parent/output"

snapfiles = cmf.utils.get_snapshots_in_dir(snapdir)

t = np.full(len(snapfiles), np.nan, dtype=float)
sep = np.full_like(t, np.nan)

for i, snapfile in enumerate(snapfiles):
    print(f"snapshot {i}")
    snap = pygad.Snapshot(snapfile, physical=True)
    t[i] = cmf.general.convert_gadget_time(snap)
    sep[i] = pygad.utils.geo.dist(snap.bh["pos"][0,:], snap.bh["pos"][1,:])
    pygad.gc_full_collect()
    snap.delete_blocks()
    del snap

plt.plot(t, sep, "-o")
plt.show()