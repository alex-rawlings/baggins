import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.check_backend()

snap = pygad.Snapshot("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_005.hdf5", physical=True)
#bgs.analysis.basic_snapshot_centring(snap)
pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

G = pygad.UnitScalar(4.3009e-6, "kpc/Msol*(km/s)**2")

idx = np.argsort(snap["r"])
print(snap["mass"][idx][:20])
print(snap.bh["pos"])
print(snap.bh["r"])

fig, ax = plt.subplots(1, 1)#, sharex="all")
#fig.set_figwidth(2.5*fig.get_figwidth())

ax.set_title("Potential")
ax.set_xlabel("r/kpc")
ax.set_ylabel(f"-pot/{snap['pot'].units}")
ax.set_xscale("log")
x = pygad.UnitArr(np.geomspace(1e-3, 1, 1000), units=snap.bh["r"].units)

axins = ax.inset_axes([0.5, 0.6, 0.45, 0.35])
for axi in (ax,):
    pot_bh = pygad.UnitArr(G*snap.bh["mass"]/np.abs(x), units=snap["pot"].units)
    axi.semilogy(snap["r"][idx], -snap["pot"][idx], marker=".")
    axi.plot(x+snap.bh["r"][0], pot_bh)
axins.set_xlim(7.5, 8)
axins.set_ylim(1e6, 1e7)
ax.indicate_inset_zoom(axins, ec="k")

'''
# this is not useful
ax[1].set_title("Binding energy (gal.)")
ax[1].loglog(snap["r"][idx][::stride], bgs.analysis.binding_energy(snap)[idx][::stride])
ax[1].set_xlabel("r/kpc")
ax[1].set_ylabel("Binding energy (rel. galaxy)")

# recentre on the BH
pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)
idx = np.argsort(snap["r"])
ax[2].set_title("Binding energy (BH)")
ax[2].loglog(snap["r"][idx][::stride], bgs.analysis.binding_energy(snap)[idx][::stride])
ax[2].set_xlabel("r/kpc")
ax[2].set_ylabel("Binding energy (rel. BH)")'''
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/potential.png"))