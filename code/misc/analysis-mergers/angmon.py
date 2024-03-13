import os.path
import numpy as np
import pygad
import matplotlib.pyplot as plt
import baggins as bgs

centering = True
pidx = "007"
snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/{pidx}/output/AC_perturb_{pidx}_017.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

if False:
    print(snap.stars["angmom"][0,:])
    snap["pos"] -= pygad.UnitArr([1,10,-20], units="kpc")
    print(snap.stars["angmom"][0,:])
    print(snap.stars["pot"])

if False:
    print(snap.stars["angmom"].min())

    print(len(snap.stars))
    J_lc = bgs.analysis.loss_cone_angular_momentum(snap, pygad.UnitScalar(10, "pc"))
    print(J_lc)
    b = snap.stars["angmom"] < J_lc
    print(np.sum(b))

if False:
    bhb = bgs.analysis.BHBinary("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AC/AC-030-0050.py", perturbID=pidx, apfile=os.path.join(bgs.HOME, "projects/collisionless-merger-sample/parameters/parameters-analysis/datacubes.py"))
    #xcom = bgs.analysis.get_com_of_each_galaxy(snap)
    #vcom = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcom)
    idx = 0
    xmin = 99
    xmax = -99
    ymin = 99
    ymax = -99
    while True:
        print(f"running {idx:03d}...")
        try:
            snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/{pidx}/output/AC_perturb_{pidx}_{idx:03d}.hdf5"
            snap = pygad.Snapshot(snapfile, physical=True)
        except:
            break

        snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
        snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")

        t = bhb.time_offset+bgs.general.convert_gadget_time(snap, new_unit='Myr')

        cols = ["tab:blue", "k"]
        marker = [",", "o"]

        fig = plt.figure()
        ax = [plt.subplot(221), plt.subplot(223), plt.subplot(122)]
        bhb.plot(ax=ax)
        ax[0].axvline(t, c="k", ls=":", lw=0.7)
        ax[1].axvline(t, c="k", ls=":", lw=0.7)

        for i, fam in enumerate(("stars", "bh")):
            subsnap = getattr(snap, fam)
            b = -subsnap["mass"] * subsnap["pot"] - 0.5 * subsnap["mass"] * pygad.utils.geo.dist(subsnap["vel"])**2
            J = pygad.utils.geo.dist(subsnap["angmom"])

            
            """if b.min() < xmin: xmin = b.min()
            if b.max() > xmax: xmax = b.max()
            if J.min() < ymin: ymin = J.min()
            if J.max() > ymax: ymax = J.max()"""
            ax[2].plot(b, J, marker=marker[i], c=cols[i], ls="")
        ax[2].set_xlim(-6e16, 6e16)
        ax[2].set_ylim(0, 2.4e12)
        ax[2].set_xlabel(r"$\mathcal{E}/m=-v^2/2 - \Phi(r)$")
        ax[2].set_ylabel("| J |")
        ax[2].set_yscale("symlog")
        ax[2].set_xscale("symlog")
        ax[2].set_title(f"{t:.2f} Myr")
        ax[2].set_xticks([-1e12, -1e8, -1e4, 0, 1e4, 1e8, 1e12])
        snap.delete_blocks()
        plt.tight_layout()
        plt.savefig(os.path.join(bgs.FIGDIR, f"analysis-explore/ang-mom/AC-030-0050-{idx:03d}.png"))
        plt.close()
        idx += 1
    print(f"xmin: {xmin:.3e}")
    print(f"xmax: {xmax:.3e}")
    print(f"ymin: {ymin:.3e}")
    print(f"ymax: {ymax:.3e}")


if True:
    bhb = bgs.analysis.BHBinary("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AC/AC-030-0050.py", perturbID=pidx, apfile=os.path.join(bgs.HOME, "projects/collisionless-merger-sample/parameters/parameters-analysis/datacubes.py"))
    #xcom = bgs.analysis.get_com_of_each_galaxy(snap)
    #vcom = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcom)
    idx = 0
    t = []
    b0 = []
    b1 = []
    while True:
        print(f"running {idx:03d}...")
        try:
            snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/{pidx}/output/AC_perturb_{pidx}_{idx:03d}.hdf5"
            snap = pygad.Snapshot(snapfile, physical=True)
        except:
            break

        snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
        snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")

        t.append(bhb.time_offset+bgs.general.convert_gadget_time(snap, new_unit='Myr'))
        subsnap = getattr(snap, "bh")
        btemp = -subsnap["mass"] * subsnap["pot"] - 0.5 * subsnap["mass"] * pygad.utils.geo.dist(subsnap["vel"])**2
        b0.append(btemp[0])
        b1.append(btemp[1])
        snap.delete_blocks()
        idx += 1
    plt.plot(t, b0, label="bh-0")
    plt.plot(t, b1, label="bh-1")
    plt.legend()
    plt.xlabel("t/Myr")
    plt.ylabel(r"$\mathcal{E}/m=-v^2/2 - \Phi(r)$")
    plt.savefig(os.path.join(bgs.FIGDIR, f"analysis-explore/binding-energy-AC-030-0050-{idx:03d}.png"))
