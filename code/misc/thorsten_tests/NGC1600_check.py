import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


snapfile = "/scratch/pjohanss/arawling/testing/NGC1600/antti-1.5-bh-1-merger_041.hdf5"
redges = np.geomspace(0.1, 20, 21)

snap = pygad.Snapshot(snapfile, physical=True)

for fam in snap.families():
    m = getattr(snap, fam)["mass"]
    print(f"{fam}: {m[0]:.3e} (# {len(m):.3e})")

'''Re, vsigRe2, vsigR2, Sigma = cmf.analysis.projected_quantities(snap, r_edges=redges, family="lowres", obs=2)


S = pygad.UnitArr(list(Sigma.values())[0], units="Msol/kpc**2")
mu = np.nanmedian(pygad.UnitArr(S, units="Msol/pc**2"), axis=0)/4.0
print(mu.units)

fig1, ax1 = plt.subplots(1,1)
ax1.loglog(cmf.mathematics.get_histogram_bin_centres(redges),mu, "-o")
ax1.set_xlabel("r/kpc")
ax1.set_ylabel(r"$\mu$/ (L$_\odot$ pc$^{-2}$)")
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "other_tests/thorsten/NGC1600-antti-check-density.png"), fig1)
'''
xcom = cmf.analysis.get_com_of_each_galaxy(snap, method="ss", family="lowres")
vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom=xcom, family="lowres")

beta, nperbin = cmf.analysis.velocity_anisotropy(snap.lowres, redges, xcom=list(xcom.values())[0], vcom=list(vcom.values())[0])
fig2, ax2 = plt.subplots(1,1)
ax2.semilogx(cmf.mathematics.get_histogram_bin_centres(redges), beta)
ax2.set_xlabel("r/kpc")
ax2.set_ylabel(r"$\beta(r)$")
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "other_tests/thorsten/NGC1600-antti-check-beta.png"), fig2)


plt.show()