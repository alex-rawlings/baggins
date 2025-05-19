import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pygad
import baggins as bgs


snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/trail_blazer/H_2M_b-H_2M_c-30.000-2.000/output")

fig, ax = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(6, 3.5))
for axi, s in zip(ax, (6, 50)):
    snap = pygad.Snapshot(snapfiles[s], physical=True)
    pygad.Translation(-pygad.analysis.center_of_mass(snap.bh)).apply(snap, total=True)
    pygad.plotting.image(snap.stars, qty="mass", cmap="cmr.eclipse", showcbar=False, scaleind="line", ax=axi, extent=100, xaxis=0, yaxis=2)
    axi.set_facecolor("k")
    axins = axi.inset_axes(
            [0.78, 0.05, 0.2, 0.2],
            xticklabels=[],
            yticklabels=[],
        )
    axins.set_xticks([])
    axins.set_yticks([])
    star_mask = pygad.BallMask(1.05 * np.max(snap.bh["r"]))
    axins.scatter(snap.bh["pos"][:, 0], snap.bh["pos"][:, 2], c="k", marker="o", ec="w", lw=0.2, zorder=0.5)
    xlim = axins.get_xlim()
    ylim = axins.get_ylim()
    axins.scatter(snap.stars[star_mask]["pos"][:, 0], snap.stars[star_mask]["pos"][:, 2], c="gold", marker="*", ec="w", lw=0.2, s=3)
    axins.set_xlim(*[1.1*xl for xl in xlim])
    axins.set_ylim(*[1.1*yl for yl in ylim])
    #axins.set_aspect("equal")
    axins.set_facecolor("k")
    axi.indicate_inset_zoom(axins, alpha=1, edgecolor="w", lw=0.5)
circ = Circle((0.4, 0.4), 0.2, transform=axins.transAxes, facecolor="white", edgecolor="w")
#axins.set_clip_path(circ)
#axins.add_patch(circ)
#axins.set_frame_on(False)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "pr_plots/merger_overview.png"), force_ext=True)
