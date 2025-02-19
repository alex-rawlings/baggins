# XXX code to make SED plot, based off the code from:
# https://github.com/astropy/SPISEA/blob/main/docs/paper_examples/Figure%207.ipynb

from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import numpy as np
import os
import matplotlib.pyplot as plt


# We'll use the UnresolvedCluster object to create the unresolved spectrum of 
# two 10^4 M_sun clusters, one at 10 Myr and the other at 1 Gyr. We'll assume 
# solar metallicity and a Kroupa IMF for both.

# Isochrone Parameters
logAge_young = 7.0 
AKs_young = 0
dist_young = 3.5e6 # M82 distance
metallicity_young = 0 

logAge_old = 9.0
AKs_old = AKs_young
dist_old = dist_young
metallicity_old = metallicity_young 

evo = evolution.Parsec()
atm_func = atmospheres.get_merged_atmosphere

# Make isochrones. We will use Isochrone objects here, since we don't need the 
# synthetic photometry (only the stellar spectra). Note that these files are 
# not saved, and will be generated each time 
iso_young = synthetic.Isochrone(logAge_young, AKs_young, dist_young, 
                                    metallicity=metallicity_young,
                                    evo_model=evo, atm_func=atm_func,
                                    mass_sampling=2)

iso_old = synthetic.Isochrone(logAge_old, AKs_old, dist_old, 
                                    metallicity=metallicity_old,
                                    evo_model=evo, atm_func=atm_func,
                                    mass_sampling=2)

# Make the unresolved clusters. This is slower than ResolvedCluster; about 30s each.
# Assign each cluster a Kroupa+2001 IMF w/o multiplicity
imf_kroupa = imf.Kroupa_2001(multiplicity=None)
clust_m1 = 5e3
clust_m2 = 5e4

clust_young_1 = synthetic.UnresolvedCluster(iso_young, imf_kroupa, clust_m1)
clust_young_2 = synthetic.UnresolvedCluster(iso_young, imf_kroupa, clust_m2)
clust_old_1 = synthetic.UnresolvedCluster(iso_old, imf_kroupa, clust_m1)
clust_old_2 = synthetic.UnresolvedCluster(iso_old, imf_kroupa, clust_m2)

# plot the unresolved cluster spectra
fig, ax = plt.subplots()
for cluster, label in zip(
    (clust_young_1, clust_young_2, clust_old_1, clust_old_2),
    ("young, low-mass", "young, high-mass", "old, low-mass", "old, high-mass")
):
    ax.semilogy(
        cluster.wave_trim * 1e-4,
        cluster.wave_trim * cluster.spec_trim,
        label=label,
        lw=2
    )

ax.set_xlim(1e-2, 5.0)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda / (\mu\mathrm{m})$")
ax.set_ylabel(r"$\log(\lambda\mathrm{F}_\lambda)$")
ax.legend()
plt.savefig("cluster_unresolved.png")