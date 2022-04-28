import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad

mdict = {"linewidth":0.5, "edgecolor":"k"}

data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05"
pidx = 51
merged = [1, 2, 4, 7, 9]

fig, ax = plt.subplots(4,2, figsize=(6, 10))
ax[0,0].set_title("Position")
ax[0,1].set_title("Velocity")
ax[0,0].set_xlabel("x/kpc")
ax[0,0].set_ylabel("y/kpc")
ax[1,0].set_xlabel("x/kpc")
ax[1,0].set_ylabel("z/kpc")
ax[2,0].set_xlabel("Child")
ax[2,0].set_ylabel(r"|BH$_\mathrm{parent}-$BH$_\mathrm{child}$|/kpc")
ax[3,0].set_xlabel("Child")
ax[3,0].set_ylabel(r"|COM$_\mathrm{SS}-$BH$_\mathrm{child}$|/kpc")
ax[0,1].set_xlabel("vx/km/s")
ax[0,1].set_ylabel("vy/km/s")
ax[1,1].set_xlabel("vx/km/s")
ax[1,1].set_ylabel("vz/km/s")
ax[2,1].set_xlabel("Child")
ax[2,1].set_ylabel(r"|BH$_\mathrm{parent}-$BH$_\mathrm{child}$|/km/s")
ax[3,1].set_xlabel("Child")
ax[3,1].set_ylabel(r"|COM$_\mathrm{SS}-$BH$_\mathrm{child}$|/km/s")

cols = cmf.plotting.mplColours()

for i in range(-1, 10):
    print(i)
    if i < 0:
        snaplist = cmf.utils.get_snapshots_in_dir(os.path.join(data_path, "output"))
        snap = pygad.Snapshot(snaplist[pidx], physical=True)
        ppos = dict.fromkeys(snap.bh["ID"])
        pvel = dict.fromkeys(snap.bh["ID"])
        for k in ppos.keys():
            idmask = pygad.IDMask(k)
            ppos[k] = snap.bh[idmask]["pos"]
            pvel[k] = snap.bh[idmask]["vel"]
    else:
        snaplist = cmf.utils.get_snapshots_in_dir(os.path.join(data_path, f"perturbations/{i:03d}/output"))
        snap = pygad.Snapshot(snaplist[0], physical=True)
    
    for j, (x,y) in enumerate(zip((0,0), (1,2))):
        ax[j,0].scatter(snap.bh["pos"][:,x], snap.bh["pos"][:,y], c=("k" if i<0 else cols[i]), **mdict)
        ax[j,1].scatter(snap.bh["vel"][:,x], snap.bh["vel"][:,y], c=("k" if i<0 else cols[i]), **mdict)
    
    if i >=0:
        alpha = 1 if i in merged else 0.3
        for j, k in enumerate(ppos.keys()):
            idmask = pygad.IDMask(k)
            print(ppos[k])
            print(snap.bh[idmask]["pos"])
            print(pvel[k])
            print(snap.bh[idmask]["vel"])
            print()
            psep = cmf.mathematics.radial_separation(snap.bh[idmask]["pos"], ppos[k])
            ax[2,0].scatter(i, psep, marker=("o" if j<1 else "s"), c=cols[i], alpha=alpha, **mdict)
            vsep = cmf.mathematics.radial_separation(snap.bh[idmask]["vel"], pvel[k])
            ax[2,1].scatter(i, vsep, marker=("o" if j<1 else "s"), c=cols[i], alpha=alpha, **mdict)
            comxsep = cmf.mathematics.radial_separation(snap.bh[idmask]["pos"], xcoms[k])
            ax[3,0].scatter(i, comxsep, c=cols[i], marker=("o" if j<1 else "s"), alpha=alpha, **mdict)
            comvsep = cmf.mathematics.radial_separation(snap.bh[idmask]["vel"], vcoms[k])
            ax[3,1].scatter(i, comvsep, c=cols[i], marker=("o" if j<1 else "s"), alpha=alpha, **mdict)


    else:
        id_masks = cmf.analysis.get_all_id_masks(snap)
        xcoms = cmf.analysis.get_com_of_each_galaxy(snap, method="ss", masks=id_masks)
        vcoms = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcoms, masks=id_masks)
        for ii, k in enumerate(xcoms.keys()):
            for j, (x,y) in enumerate(zip((0,0), (1,2))):
                ax[j,0].scatter(xcoms[k][x], xcoms[k][y], marker="x", c="k")
                ax[j,1].scatter(vcoms[k][x], vcoms[k][y], marker="x", c="k")
            bhidmask = pygad.IDMask(k)
            parent_xsep = pygad.utils.geo.dist(snap.bh[bhidmask]["pos"], xcoms[k])
            parent_vsep = pygad.utils.geo.dist(snap.bh[bhidmask]["vel"], vcoms[k])
            ax[3,0].axhline(parent_xsep, c="k", alpha=0.5, ls=":", label=("Parent Sep." if ii==0 else ""))
            ax[3,1].axhline(parent_vsep, c="k", alpha=0.5, ls=":")
    '''snap["pos"] -= list(xcoms.values())[0]
    snap["vel"] -= list(vcoms.values())[0]
    H, KE, PE = cmf.analysis.calculate_Hamiltonian(snap)
    sc = ax.scatter(KE-PE, KE, c=("k" if i<0 else cols[i]))
    ax.scatter(KE-PE, PE, c=sc.get_facecolor())'''

    snap.delete_blocks()
ax[3,0].legend()
plt.show()