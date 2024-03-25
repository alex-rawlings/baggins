import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad

paramfile_base = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/resolution-convergence/hardening/"

fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$N_\star$")
ax.set_ylabel(r"(K$_\mathrm{c}$+U$_\mathrm{c}$)/(K$_\mathrm{p}$+U$_\mathrm{p}$)")


for pfR, n, col in zip(("DE-030-0005r1.py", "DE-030-0005r2.py", "DE-030-0005r5.py"), ("1", "2", "5"), ("tab:blue", "tab:orange", "tab:green")):
    pf = os.path.join(paramfile_base, pfR)
    pfv = bgs.utils.read_parameters(pf)
    #get the parent
    snaplist = bgs.utils.get_snapshots_in_dir(os.path.join(pfv.full_save_location, "output"))
    if len(snaplist) > 1:
        snap_perturb_idx = bgs.analysis.snap_num_for_time(snaplist, pfv.perturbTime*1e3)
    else:
        snap_perturb_idx = -1
    snap = pygad.Snapshot(snaplist[snap_perturb_idx], physical=True)
    total_energy_p = bgs.analysis.calculate_Hamiltonian(snap.stars)
    #ax.scatter("Parent", total_energy_p, c=col, label=r"$N_\mathrm{{fid}}$/{}".format(n))
    #get the children
    for i in range(10):
        print("Child {}".format(i))
        perturb_idx = "{:03d}".format(i)
        snaplist = bgs.utils.get_snapshots_in_dir(os.path.join(pfv.full_save_location, pfv.perturbSubDir, perturb_idx, "output"))
        idx = min(5, len(snaplist)-1)
        snap = pygad.Snapshot(snaplist[idx], physical=True)
        total_energy = bgs.analysis.calculate_Hamiltonian(snap.stars)
        ax.scatter(len(snap.stars["mass"]), total_energy/total_energy_p, c=col, label=(r"$N_\mathrm{{fid}}$/{}".format(n) if i==0 else ""))
        snap.delete_blocks()
ax.legend()
plt.show()