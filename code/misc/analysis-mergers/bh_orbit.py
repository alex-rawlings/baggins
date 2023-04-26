import os.path
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf


pathnum = 0
compare_e = True

kpc = ketjugw.units.pc * 1e3

datapath = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_05",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_07",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_09",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_099"
]

e_vals = [0.5, 0.7, 0.9, 0.99]


def plotter(p, ax=None, compare_e=False, label=None):
    bhfiles = cmf.utils.get_ketjubhs_in_dir(p)

    if ax is None:
        fig, ax = plt.subplots(2,1, sharex="all", sharey="row")
        ax[0].set_ylabel("z/kpc")
        ax[1].set_xlabel("x/kpc")
        ax[1].set_ylabel("y/kpc")

    for i, bhf in enumerate(bhfiles):
        bh1, bh2, *_ = cmf.analysis.get_bh_particles(bhf)
        if label is None:
            label = i
        l = ax[0].plot(bh1.x[:,0]/kpc, bh1.x[:,2]/kpc)
        ax[1].plot(bh1.x[:,0]/kpc, bh1.x[:,1]/kpc, c=l[0].get_color(), label=label)
        ax[0].plot(bh2.x[:,0]/kpc, bh2.x[:,2]/kpc, c=l[0].get_color(), ls=":")
        ax[1].plot(bh2.x[:,0]/kpc, bh2.x[:,1]/kpc, c=l[0].get_color(), ls=":")
        if compare_e: break
    return ax


if compare_e:
    ax = None
    for d, l in zip(datapath, e_vals):
        ax = plotter(d, ax=ax, compare_e=compare_e, label=l)
else:
    ax = plotter(datapath[pathnum])

ax[-1].legend()
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "merger/gualandris_ics.png"))
plt.show()
