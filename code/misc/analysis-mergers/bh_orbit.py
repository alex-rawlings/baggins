import os.path
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf


datapath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_05"

bhfiles = cmf.utils.get_ketjubhs_in_dir(datapath)

kpc = ketjugw.units.pc * 1e3

fig, ax = plt.subplots(2, len(bhfiles), sharex="all", sharey="all")
ax[0,0].set_ylabel("z/kpc")
for axi in ax[1,:]: axi.set_xlabel("x/kpc")
ax[1,0].set_ylabel("y/kpc")

for i, bhf in enumerate(bhfiles):
    bh1, bh2, *_ = cmf.analysis.get_bh_particles(bhf)
    ax[0,i].set_title(f"Realisation {i}")
    l = ax[0,i].plot(bh1.x[:,0]/kpc, bh1.x[:,2]/kpc)
    ax[1,i].plot(bh1.x[:,0]/kpc, bh1.x[:,1]/kpc, c=l[0].get_color())
    ax[0,i].plot(bh2.x[:,0]/kpc, bh2.x[:,2]/kpc, c=l[0].get_color(), ls=":")
    ax[1,i].plot(bh2.x[:,0]/kpc, bh2.x[:,1]/kpc, c=l[0].get_color(), ls=":")

cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "merger/gualandris_ics.png"))
plt.show()