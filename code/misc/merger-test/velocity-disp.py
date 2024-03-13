import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import baggins as bgs
import pygad

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.1/perturbations/003/output/AC_perturb_003_005.hdf5"
a = 1e-3

snap = pygad.Snapshot(snapfile, physical=True)
if False:
    mask = np.logical_and(snap.stars["vel"][:,0] > np.nanquantile(snap.stars["vel"][:,0], 1e-3), snap.stars["vel"][:,0] < np.nanquantile(snap.stars["vel"][:,0], 1-1e-3))
    masked_vel = snap.stars["vel"][:,0][mask]
    pars = scipy.stats.gennorm.fit(masked_vel)
    print(pars)
    rv = scipy.stats.gennorm(*pars)
    x = np.linspace(rv.ppf(a), rv.ppf(1-a), 500)
    plt.hist(masked_vel, 100, alpha=0.6, density=True)
    plt.plot(x, rv.pdf(x), c="tab:red")
    plt.show()
else:
    reps = 10
    means = np.full(reps, np.nan)
    for i in range(reps):
        print(i)
        Q = bgs.analysis.projected_quantities(snap, obs=4)
        print(list(Q.values())[0]["vsig"])
        means[i] = np.nanmean(list(Q.values())[0]["vsig"])
    print(means)
    plt.hist(means)
    plt.show()