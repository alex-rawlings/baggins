import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


snapfile = "/scratch/pjohanss/arawling/testing/NGC1600/antti-1.5-bh-1-merger_041.hdf5"
redges = np.geomspace(0.01, 50, 101)

snap = pygad.Snapshot(snapfile, physical=True)

for fam in snap.families():
    m = getattr(snap, fam)["mass"]
    print(f"{fam}: {m[0]:.3e} (# {len(m):.3e})")

Re, vsigRe2, vsigR2, Sigma = cmf.analysis.projected_quantities(snap, r_edges=redges, family="lowres", obs=2)


S = pygad.UnitArr(list(Sigma.values())[0], units="Msol/kpc**2")
mu = np.nanmedian(pygad.UnitArr(S, units="Msol/pc**2"), axis=0)/4.0
print(mu.units)

plt.loglog(cmf.mathematics.get_histogram_bin_centres(redges),mu, "-o")
plt.show()