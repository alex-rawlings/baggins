import os.path
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf
import pygad

datasets = [
    "/scratch/pjohanss/arawling/collisionless_merger/resolution-convergence/mergers5/D-E-3.0-0.005/perturbations",
    "/scratch/pjohanss/arawling/collisionless_merger/resolution-convergence/mergers2/D-E-3.0-0.005/perturbations",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/D-E-3.0-0.005/perturbations"
]

if False:
    H = []
    Gps = []
    e0_scatter = []

    for i, dataset in enumerate(datasets):
        H.append([])
        Gps.append([])
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
                H[i].append(bh_binary.H)
                Gps[i].append(bh_binary.G_rho_per_sigma)
                bh_binary.formation_eccentricity_spread()
                e0_scatter[i].append(bh_binary.e0_spread)
            except ValueError:
                continue

    data_dict = dict(
                    H = H,
                    Gps = Gps,
                    e0_scatter = e0_scatter
    )
    cmf.utils.save_data(data_dict, "convergence.pickle")
else:
    data_dict = cmf.utils.load_data("convergence.pickle")
    H = data_dict["H"]
    Gps = data_dict["Gps"]
    e0_scatter = data_dict["e0_scatter"]

fig, ax = plt.subplots(1,3, sharex="all")
for H_i, Gps_i, e0_scatter_i, res in zip(H, Gps, e0_scatter, (0.2,0.5, 1)):
    for i, val in enumerate((np.array(H_i) * np.array(Gps_i), H_i, e0_scatter_i)):
        med_val = np.nanmedian(val)
        yerr = np.full((2,1), np.nan, dtype=float)
        yerr[0,0], yerr[1,0] = med_val - np.nanquantile(val, 0.25, axis=-1), np.nanquantile(val, 0.75, axis=-1) - med_val
        ax[i].errorbar(res, med_val, yerr=yerr, fmt="o")
ax[1].set_xlabel("Resolution / fiducial resolution")
ax[0].set_ylabel(r"d(1/a)/dt [1/(pc * yr)]")
ax[1].set_ylabel(r"H")
ax[2].set_ylabel(r"$\sigma_{ecc}$")
plt.show()

