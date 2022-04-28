import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw


#mainpath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations"
#mainpath = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/H-H-3.0-0.05/perturbations"
mainpath = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/H-H-3.0-0.05/perturbations-gadget"

ketjufiles = cmf.utils.get_ketjubhs_in_dir(mainpath)
ax = None

for i, k in enumerate(ketjufiles):
    print(f"{i:03d}")
    bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
    op = ketjugw.orbital_parameters(bh1, bh2)
    ax = cmf.plotting.binary_param_plot(op, ax=ax, label=f"{i:03d}")
ax[0].legend()
plt.show()
