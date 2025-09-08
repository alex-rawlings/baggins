from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs


#maindir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/ecc-chaos/e-090/8M"
maindir = "/scratch/pjohanss/madleevi/emm/e-090/100K"


fig, ax = plt.subplots(figsize=(3,3))
facecol = "#303030"
ax.set_facecolor(facecol)

kfs = bgs.utils.get_ketjubhs_in_dir(maindir)
kfs = [kfs[0], kfs[1], kfs[6], kfs[7], kfs[2]]
kwargs1 = {"lw":2, "alpha":1, "markevery":[-1], "marker":"o", "mec":ax.get_facecolor(), "mew":0.1, "zorder":1, "solid_capstyle":"round"}
kwargs2 = copy(kwargs1)
cmapper1, _ = bgs.plotting.create_normed_colours(0, 1.1*len(kfs), cmap="flare")
cmapper2, _ = bgs.plotting.create_normed_colours(0, 1.1*len(kfs), cmap="crest")
peri_idx_wanted = 1

for i, kf in enumerate(kfs):
    if i>0:
        kwargs1["zorder"] = 0.5
        kwargs2["zorder"] = kwargs1["zorder"]
    bh1, bh2, _ = bgs.analysis.get_bound_binary(kf)
    bh1, bh2 = bgs.analysis.move_to_centre_of_mass(bh1, bh2)
    _, peri_idxs = bgs.analysis.find_pericentre_time(bh1, bh2)
    bh1.x /= bgs.general.units.kpc
    bh2.x /= bgs.general.units.kpc
    bh1.t /= bgs.general.units.Myr
    bh2.t /= bgs.general.units.Myr
    sli = slice(peri_idxs[peri_idx_wanted]-200, peri_idxs[peri_idx_wanted]+500)
    print(sli, f"{bh1.t[sli][0]:.1f}", kwargs1["alpha"])
    for bh, kw, cmapper in zip((bh1, bh2), (kwargs1, kwargs2), (cmapper1, cmapper2)):
        ax.plot(bh.x[sli,0], bh.x[sli,2], **kw, c=cmapper(i))

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
axlim = max(-np.diff(list(ax.get_xlim())), -np.diff(list(ax.get_ylim()))) / 1.5
print(axlim)
ax.set_xlim(-axlim, axlim)
ax.set_ylim(-axlim, axlim)
bgs.plotting.savefig("thesis_front.pdf", force_ext=True)