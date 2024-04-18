import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


fig, ax = plt.subplots(1,2)

idx = "010"

snaplist = [
    "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/output/DE_052.hdf5",
    "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations_eta_0002/000/output/DE_perturb_000_{}.hdf5".format(idx),
    "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations_eta_0005/000/output/DE_perturb_000_{}.hdf5".format(idx),
    "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations_eta_0020/000/output/DE_perturb_000_{}.hdf5".format(idx)
]

labels = ["Parent", r"$\eta=$0.002", r"$\eta=$0.005", r"$\eta=$0.020"]
cols = bgs.plotting.mplColours()

relerr = lambda h, hp: np.abs((h-hp)/hp)

for i, snapfile in enumerate(snaplist):
    snap = pygad.Snapshot(snapfile, physical=True)
    hamiltonian_all = -bgs.analysis.calculate_Hamiltonian(snap)
    hamiltonian_stars = -bgs.analysis.calculate_Hamiltonian(snap.stars)
    hamiltonian_dm = -bgs.analysis.calculate_Hamiltonian(snap.dm)
    if i==0:
        parent_H_all = hamiltonian_all
        parent_H_stars = hamiltonian_stars
        parent_H_dm = hamiltonian_dm
    else:
        ax[1].scatter(labels[i], relerr(hamiltonian_all,parent_H_all), c=cols[i])
        ax[1].scatter(labels[i], relerr(hamiltonian_stars,parent_H_stars), c=cols[i], marker="*")
        ax[1].scatter(labels[i], relerr(hamiltonian_dm,parent_H_dm), c=cols[i], marker="s")
    ax[0].scatter(labels[i], hamiltonian_all, c=cols[i], label=("All" if i==0 else ""))
    ax[0].scatter(labels[i], hamiltonian_stars, c=cols[i], marker="*", label=("Stars" if i==0 else ""))
    ax[0].scatter(labels[i], hamiltonian_dm, c=cols[i], marker="s", label=("DM" if i==0 else ""))
ax[0].legend()
ax[0].set_yscale("log")
ax[0].set_ylabel("-H")
ax[1].set_ylabel(r"|(H-H$_\mathrm{parent}$)/H$_\mathrm{parent}$|")
plt.show()