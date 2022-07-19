import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-D-3.0-1.0/perturbations/002/output/AD_perturb_002_020.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)
r_edges = np.geomspace(1e-2, 300, 101)
rng = np.random.default_rng(42)

x = cmf.mathematics.get_histogram_bin_centres(r_edges)

"""plt.axvline(0.1, c="k")
plt.loglog(x, 
            pygad.analysis.profile_dens(snap.stars, "mass", center=pygad.analysis.center_of_mass(snap.bh), r_edges=r_edges, proj=0))
plt.show()
quit()"""


print("Determining LOS quantities...")
re, vsig, Sigma = cmf.analysis.projected_quantities(snap, obs=3, r_edges=r_edges, rng=rng)

plt.axvline(0.1, c="k")
for i, (k,v) in enumerate(Sigma.items()):
    if i > 0: break
    plt.plot(x, v["estimate"])
    plt.fill_between(x, y1=v["low"], y2=v["high"], alpha=0.4)
plt.yscale("log")
plt.xscale("log")
plt.show()