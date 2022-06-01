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
    data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05"
    perturbtime = 5071
idx = 4500
idx2 = -1000
alpha = 0.4
t0 = 5071
#parentdir = os.path.join(data_path, "output")
#parentfile = cmf.utils.get_ketjubhs_in_dir(parentdir)[0]
all_list = cmf.utils.get_ketjubhs_in_dir(data_path)
#all_list = cmf.utils.get_ketjubhs_in_dir(os.path.join(data_path, "perturbations"))
#all_list.insert(0, parentfile)
#all_list.insert(0, "testing")
myr = ketjugw.units.yr * 1e6
pc = ketjugw.units.pc
kms = ketjugw.units.km_per_s

fig, ax = plt.subplots(2,2)
for i, bhfile in enumerate(all_list):
    print(bhfile)
    #if i==0: continue
    bh1, bh2, merged = cmf.analysis.get_bh_particles(bhfile)
    ls = "--" if i==0 else "-"
    if i==0:
        #continue
        '''snaplist = cmf.utils.get_snapshots_in_dir(parentdir)
        snap_idx = cmf.analysis.snap_num_for_time(snaplist, perturbtime)
        snap = pygad.Snapshot(snaplist[snap_idx], physical=True)
        t0 = cmf.general.convert_gadget_time(snap, new_unit="Myr")'''
        idx0 = np.argmax(bh1.t/myr > t0)
        idxs = np.r_[idx0-idx:idx0+idx2]
        for j, (x,y) in enumerate(zip((0,0), (1,2))):
            #ax[j,0].scatter(snap.bh["pos"][0,x]*1e3, snap.bh["pos"][0,y]*1e3, marker="s", s=50, c="k", alpha=alpha, label=("Perturbation Applied" if j==0 else ""))
            #ax[j,0].scatter(snap.bh["pos"][1,x]*1e3, snap.bh["pos"][1,y]*1e3, marker="s", c="k", s=50, alpha=alpha)
            ax[j,0].plot(bh1.x[idxs,x]/pc, bh1.x[idxs,y]/pc, ls=ls, c="k", label=("Parent" if j==0 else ""), marker="o", markevery=[-1])
            ax[j,0].plot(bh2.x[idxs,x]/pc, bh2.x[idxs,y]/pc, ls=ls, c="k", marker="o", markevery=[-1])

            #ax[j,1].scatter(snap.bh["vel"][0,x]*1e3, snap.bh["vel"][0,y]*1e3, marker="s", s=50, c="k", alpha=alpha)
            #ax[j,1].scatter(snap.bh["vel"][1,x]*1e3, snap.bh["vel"][1,y]*1e3, marker="s", c="k", s=50, alpha=alpha)
            ax[j,1].plot(bh1.v[idxs,x]/kms, bh1.v[idxs,y]/kms, ls=ls, c="k")
            ax[j,1].plot(bh2.v[idxs,x]/kms, bh2.v[idxs,y]/kms, ls=ls, c="k")
    else:
        for j, (x,y) in enumerate(zip((0,0), (1,2))):
            l = ax[j,0].plot(bh1.x[:idx,x]/pc, bh1.x[:idx,y]/pc, ls=ls, markevery=[-1], marker="o", label=("Child" if j==0 and i==1 else ""))
            ax[j,0].plot(bh2.x[:idx,x]/pc, bh2.x[:idx,y]/pc, ls=ls, markevery=[-1], c=l[0].get_color(), marker="o")

            ax[j,1].plot(bh1.v[:idx,x]/kms, bh1.v[:idx,y]/kms, ls=ls, markevery=[-1], c=l[0].get_color(), marker="o")
            ax[j,1].plot(bh2.v[:idx,x]/kms, bh2.v[:idx,y]/kms, ls=ls, markevery=[-1], c=l[0].get_color(), marker="o")
ax[0,0].legend()
ax[1,0].set_xlabel("x/pc")
ax[0,0].set_ylabel("z/pc")
ax[1,0].set_ylabel("y/pc")
ax[1,1].set_xlabel("vx/km/s")
ax[0,1].set_ylabel("vz/km/s")
ax[1,1].set_ylabel("vy/km/s")
plt.suptitle(r"$t\approx {:.3f}$ Gyr".format(perturbtime/1e3))
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "merger-test/AC-030-0050-perturbs.png"))
plt.show()