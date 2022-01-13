import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw
import pygad

if False:
    data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/D-E-3.0-0.05"
    perturbtime = 4836
else:
    data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/C-E-3.0-0.005"
    perturbtime = 7160
idx = 800
idx2 = 500
alpha = 0.4
parentdir = os.path.join(data_path, "output")
parentfile = cmf.utils.get_ketjubhs_in_dir(parentdir)[0]
all_list = cmf.utils.get_ketjubhs_in_dir(os.path.join(data_path, "perturbations"))
all_list.insert(0, parentfile)
myr = ketjugw.units.yr * 1e6
pc = ketjugw.units.pc

fig, ax = plt.subplots(1,2)
for i, bhfile in enumerate(all_list):
    print(bhfile)
    bh1, bh2, merged = cmf.analysis.get_bh_particles(bhfile)
    ls = "--" if i==0 else "-"
    if i==0:
        snaplist = cmf.utils.get_snapshots_in_dir(parentdir)
        snap_idx = cmf.analysis.snap_num_for_time(snaplist, perturbtime)
        snap = pygad.Snapshot(snaplist[snap_idx], physical=True)
        t0 = cmf.general.convert_gadget_time(snap, new_unit="Myr")
        idx0 = np.argmax(bh1.t/myr > t0)+1
        for j, (x,y) in enumerate(zip((0,2), (0,1))):
            ax[j].scatter(snap.bh["pos"][0,x]*1e3, snap.bh["pos"][0,y]*1e3, marker="s", s=50, c="k", alpha=alpha, label=("Perturbation Applied" if j==0 else ""))
            ax[j].scatter(snap.bh["pos"][1,x]*1e3, snap.bh["pos"][1,y]*1e3, marker="s", c="k", s=50, alpha=alpha)
            ax[j].plot(bh1.x[idx0-idx:idx0+idx2,x]/pc, bh1.x[idx0-idx:idx0+idx2,y]/pc, ls=ls, c="k", label=("Parent" if j==0 else ""))
            ax[j].plot(bh2.x[idx0-idx:idx0+idx2,x]/pc, bh2.x[idx0-idx:idx0+idx2,y]/pc, ls=ls, c="k")
    else:
        for j, (x,y) in enumerate(zip((0,2), (0,1))):
            l = ax[j].plot(bh1.x[:idx,x]/pc, bh1.x[:idx,y]/pc, ls=ls, markevery=[-1], marker="o", label=("Child" if j==0 and i==1 else ""))
            ax[j].plot(bh2.x[:idx,x]/pc, bh2.x[:idx,y]/pc, ls=ls, markevery=[-1], c=l[0].get_color(), marker="o")
ax[0].legend()
ax[0].set_xlabel("x/pc")
ax[0].set_ylabel("z/pc")
ax[1].set_xlabel("x/pc")
ax[1].set_ylabel("y/pc")
plt.suptitle(r"$t\approx {:.3f}$ Gyr".format(perturbtime/1e3))
plt.show()