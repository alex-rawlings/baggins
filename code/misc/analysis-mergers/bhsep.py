import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cm_functions as cmf
from ketjugw.units import yr, pc, km_per_s

myr = 1e6 * yr
kpc = 1e3 * pc


def mass_wgt_mean(m1, q1, m2, q2):
    m1 = m1[:,np.newaxis]
    m2 = m2[:,np.newaxis]
    return (m1*q1 + m2*q2) / (m1+m2)

#mainpath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations"
mainpath = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/H-H-3.0-0.05/perturbations-gadget-3.5pc"

ketjufiles = cmf.utils.get_ketjubhs_in_dir(mainpath)

fig, ax = plt.subplots(10, 6, sharex="col", sharey="col", figsize=(8,8))
#fig2, ax2 = plt.subplots(6, 10, sharex="row", sharey="row", figsize=(15,8))
#fig3, ax3 = plt.subplots(6, 10, sharex="row", sharey="row", figsize=(15,8))

for axi in ax[:,0]: axi.set_ylabel("y/kpc")
ax[-1,0].set_xlabel("x/kpc")
for axi in ax[:,1]: axi.set_ylabel("z/kpc")
ax[-1,1].set_xlabel("x/kpc")
for axi in ax[:,2]: axi.set_ylabel("|x|/kpc")
ax[-1,2].set_xlabel("t/Myr")
for axi in ax[:,3]: axi.set_ylabel("vy/km/s")
ax[-1,3].set_xlabel("vx/km/s")
for axi in ax[:,4]: axi.set_ylabel("vz/km/s")
ax[-1,4].set_xlabel("vx/km/s")
for axi in ax[:,5]: axi.set_ylabel("|v|/km/s")
ax[-1,5].set_xlabel("t/Myr")


for i, k in enumerate(ketjufiles):
    print(f"{i:03d}")
    bh1, bh2, merged = cmf.analysis.get_bh_particles(k)
    bh1bound, bh2bound, _ = cmf.analysis.get_bound_binary(k)

    xcom = mass_wgt_mean(bh1.m, bh1.x, bh2.m, bh2.x)
    vcom = mass_wgt_mean(bh1.m, bh1.v, bh2.m, bh2.v)

    bh1x = bh1.x - xcom
    bh2x = bh2.x - xcom 
    bh1v = bh1.v - vcom
    bh2v = bh2.v - vcom

    bh1pre_mask = bh1.t < 50*myr#bh1.t < bh1bound.t[0]
    bh2pre_mask = bh2.t < 50*myr#bh2.t < bh2bound.t[0]

    for j, (x,y) in enumerate(zip((0,0), (1,2))):
        l, = ax[i,j].plot(bh1x[bh1pre_mask, x]/kpc, bh1x[bh1pre_mask, y]/kpc, markevery=[-1], marker=".")
        ax[i,j].plot(bh2x[bh2pre_mask, x]/kpc, bh2x[bh2pre_mask, y]/kpc, markevery=[-1], marker=".")
        l, = ax[i,j+3].plot(bh1v[bh1pre_mask, x]/km_per_s, bh1v[bh1pre_mask, y]/km_per_s, markevery=[-1], marker=".")
        ax[i,j+3].plot(bh2v[bh2pre_mask, x]/km_per_s, bh2v[bh2pre_mask, y]/km_per_s, markevery=[-1], marker=".")

    
    sep = cmf.mathematics.radial_separation(bh1x[bh1pre_mask, :], bh2x[bh2pre_mask,:])
    #peaks, props = scipy.signal.find_peaks(-sep, height=(-1e16, None))
    #print(len(peaks))
    #ax.scatter(bh1.t[bh1pre_mask][peaks]/myr, sep[peaks])
    ax[i,2].semilogy(bh1.t[bh1pre_mask]/myr, sep/kpc)

    sep = cmf.mathematics.radial_separation(bh1v[bh1pre_mask, :], bh2v[bh2pre_mask,:])
    ax[i,5].semilogy(bh1.t[bh1pre_mask]/myr, sep/km_per_s)

    #ax[0,i].set_title(f"{i:03d}")
    
    """for j in range(3):
        ax2[j,i].plot(bh1.t[bh1pre_mask]/myr, bh1x[bh1pre_mask, j]/kpc, markevery=[-1], marker=".")
        ax2[j,i].plot(bh2.t[bh2pre_mask]/myr, bh2x[bh1pre_mask, j]/kpc, markevery=[-1], marker=".")
        ax2[j+3,i].plot(bh1.t[bh1pre_mask]/myr, bh1v[bh1pre_mask, j]/km_per_s, markevery=[-1], marker=".")
        ax2[j+3,i].plot(bh2.t[bh2pre_mask]/myr, bh2v[bh2pre_mask, j]/km_per_s, markevery=[-1], marker=".")

        ax3[j,i].plot(bh1.t[bh1pre_mask]/myr, (bh1x[bh1pre_mask, j]-bh2x[bh2pre_mask, j])/kpc, markevery=[-1], marker=".")
        ax3[j+3,i].plot(bh1.t[bh1pre_mask]/myr, (bh1v[bh1pre_mask, j]-bh2v[bh2pre_mask, j])/km_per_s, markevery=[-1], marker=".")
    
    ax2[0,i].set_title(f"{i:03d}")
    ax3[0,i].set_title(f"{i:03d}")"""


plt.show()