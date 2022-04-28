import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


mainpath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations"
#mainpath = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/H-H-3.0-0.05/perturbations"

ketjufiles = cmf.utils.get_ketjubhs_in_dir(mainpath)
fig, ax = plt.subplots(1,1)

for i, k in enumerate(ketjufiles):
    label = f"{i:03d}"
    print(label)
    bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
    e = -ketjugw.orbital_energy(bh1, bh2)
    op = ketjugw.orbital_parameters(bh1, bh2)
    alpha = 0.9 if merged() else 0.3
    ax.loglog(op["e_t"], e, label=label, alpha=alpha)

    """bh1, bh2, merged = cmf.analysis.get_bh_particles(k)
    e = ketjugw.orbital_energy(bh1, bh2)
    bound_idx = np.argmax(e<0)
    alpha = 0.9 if merged() else 0.3
    ax.loglog(bh1.t, np.abs(e), label=f"{i:03d}", markevery=[bound_idx], marker="o", alpha=alpha)"""
ax.legend()
plt.show()
