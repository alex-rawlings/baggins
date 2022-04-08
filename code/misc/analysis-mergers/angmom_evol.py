import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad


snapdir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/002/output"

bhfile = os.path.join(snapdir, "ketju_bhs_cp.hdf5")
snaplist = cmf.utils.get_snapshots_in_dir(snapdir)

t = np.full_like(snaplist, np.nan, dtype=float)
r = np.full_like(t, np.nan)
theta = np.full_like(t, np.nan)
angmom_stars = np.full((len(snaplist),3), np.nan, dtype=float)
angmom_bh = np.full_like(angmom_stars, np.nan)

fig, ax = plt.subplots(3,1)
for i, snapfile in enumerate(snaplist):
    if i>9: break
    print(i)
    snap = pygad.Snapshot(snapfile, physical=True)
    snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
    snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")
    ballmask = pygad.BallMask(pygad.UnitScalar(30, "kpc"))
    t[i] = cmf.general.convert_gadget_time(snap)
    r[i] = pygad.utils.geo.dist(snap.bh["pos"][0,:], snap.bh["pos"][1,:])
    theta[i] = cmf.analysis.angular_momentum_difference_gal_BH(snap, mask=ballmask)
    angmom_stars[i,:] = snap.stars[ballmask]["angmom"].sum(axis=0)
    angmom_bh[i,:] = snap.bh["angmom"].sum(axis=0)
    snap.delete_blocks()
    del snap
#print(angmom_bh[:10,:])
#print(angmom_stars[:10,:])
#print(theta)
for i in range(3):
    ax[i].plot(t, angmom_bh[:,i])
    ax[i].plot(t, angmom_stars[:,i])

fig, ax2 = plt.subplots(1,1)
ax2.plot(t,theta, "-o")

fig4, ax4 = plt.subplots(1,1)
ax4.plot(t, r, "-o")

fig, ax3 = plt.subplots(3,1, sharex="all")
flips = np.abs(np.diff(np.sign(theta-np.pi/2)))>1
for i in range(3):
    ax3[i].plot(t, np.sign(angmom_bh[:,i]), "-o", label="BH")
    ax3[i].plot(t, np.sign(angmom_stars[:,i]), "-o", label="stars")
    for j, flip in enumerate(flips, start=1):
        if flip:
            ax3[i].axvline(t[j], c="k", ls=":")

ax3[0].legend()
ax3[-1].set_xlabel("Time/Gyr")
for i, p in enumerate(("x", "y", "z")):
    ax3[i].set_ylabel(f"sign(L{p})")
plt.show()