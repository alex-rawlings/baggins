import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/000/output/AC_perturb_000_074.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

Re, vsig, rho = cmf.analysis.projected_quantities(snap, obs=5)

joint_rho = list(rho.values())[0]
yerr_low = joint_rho["estimate"] - joint_rho["low"]
yerr_up = joint_rho["high"] - joint_rho["estimate"]

print(joint_rho["estimate"].shape)

r = cmf.mathematics.get_histogram_bin_centres(np.geomspace(2e-1, 20, 51))
plt.errorbar(r, joint_rho["estimate"], yerr=[yerr_low, yerr_up], capsize=2, fmt="-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("r/kpc")
plt.ylabel(r"$\rho$")
plt.show()