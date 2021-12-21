import os.path
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf
import pygad

datasets = [
    "/scratch/pjohanss/arawling/collisionless_merger/resolution-convergence/mergers5/D-E-3.0-0.005/perturbations",
    #"/scratch/pjohanss/arawling/collisionless_merger/resolution-convergence/mergers2/D-E-3.0-0.005/perturbations",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/D-E-3.0-0.005/perturbations"
]

inv_da_dt = []
e0_scatter = []

for i, dataset in enumerate(datasets):
    inv_da_dt.append([])
    e0_scatter.append([])
    for j in range(10):
        print("Iteration {}".format(j))
        print("----------------")
        perturb_num = "{:03d}".format(j)
        full_data_path = os.path.join(dataset, perturb_num, "output")
        snaplist = cmf.utils.get_snapshots_in_dir(full_data_path)
        bhfile = cmf.utils.get_ketjubhs_in_dir(full_data_path)[0]
        try:
            bh_binary = cmf.analysis.BHBinary(bhfile, snaplist, 12)
            if j == -9:
                bh_binary.plot()
                plt.show()
                quit()
            bh_binary.get_influence_and_hard_radius()
            bh_binary.fit_analytic_form()
            inv_da_dt[i].append(bh_binary.H)
            bh_binary.formation_eccentricity_spread()
            e0_scatter[i].append(bh_binary.e0_spread)
        except ValueError:
            continue

fig, ax = plt.subplots(1,1, sharex="all")
for inv_da_dt_i, e0_scatter_i, res in zip(inv_da_dt, e0_scatter, (0.2,1,)):
    ax.errorbar(res, np.nanmean(inv_da_dt_i), yerr=np.nanstd(inv_da_dt_i), fmt="o")
    #ax[1].errorbar(res, np.nanmean(e0_scatter_i), yerr=np.nanstd(e0_scatter_i), fmt="o")
    #res_array = np.full_like(inv_da_dt_i, res, dtype=float)
    #ax[0].scatter(res_array, inv_da_dt_i)
    #ax[1].scatter(res_array, e0_scatter_i)
ax.set_xlabel("Resolution / fiducial resolution")
ax.set_ylabel(r"H")
#ax[1].set_ylabel(r"$\sigma_{ecc}$")
plt.show()

