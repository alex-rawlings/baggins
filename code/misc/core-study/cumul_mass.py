import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import yaml
import baggins as bgs


fig, ax = plt.subplots(1,1)

files = []
with open("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml", "r") as f:
    data = yaml.load(f, yaml.SafeLoader)
for k, v in data["snap_nums"].items():
    if v is not None and v > 0:
        files.append(os.path.join(data["parent_dir"], f'kick-vel-{k.lstrip("v")}/output/snap_{v:03d}.hdf5'))

cmapper, sm = bgs.plotting.create_normed_colours(1, 1200)

for i, f in enumerate(files):
    snap = pygad.Snapshot(f, physical=True)
    idx = np.argsort(snap.stars["r"])
    label = f.split("/")[-3]
    ax.loglog(snap.stars["r"][idx], np.cumsum(snap.stars["mass"][idx]), label=label, c=cmapper(int(label.replace("kick-vel-", ""))), lw=2)
    snap.delete_blocks()
    pygad.gc_full_collect()
#ax.legend()
plt.show()