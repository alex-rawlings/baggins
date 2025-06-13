import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
import pygad
import baggins as bgs


snapfiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/trail_blazer/H_2M_b-H_2M_c-30.000-2.000/output/snap_006.hdf5",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-1020/output/snap_010.hdf5"
]
R = 0.04

def convert_axes_coords_to_data_coords(ax, coords):
    return ax.transData.inverted().transform(ax.transAxes.transform((coords)))

def inset_plotter(bounds, snap, ax, bhi=0, ang=(80, 200)):
    axins = ax.inset_axes(
            bounds,
            xticklabels=[],
            yticklabels=[],
        )
    axins.set_xticks([])
    axins.set_yticks([])
    star_mask = pygad.BallMask(0.95 * R, center=snap.bh["pos"][bhi, :])
    print(f"Number of stars: {len(snap.stars[star_mask]):.2e}")
    axins.scatter(snap.bh["pos"][bhi, 0], snap.bh["pos"][bhi, 2], c="k", marker="o", ec="w", lw=0.2, zorder=0.5)
    axins.scatter(snap.stars[star_mask]["pos"][:, 0], snap.stars[star_mask]["pos"][:, 2], c="goldenrod", marker="*", ec="w", lw=0.05, s=3, zorder=0.1)
    axins.set_facecolor("k")
    circ = Circle((0.5, 0.5), 0.48, transform=axins.transAxes, facecolor="k", edgecolor="w", zorder=0.01)
    axins.set_clip_path(circ)
    axins.add_patch(circ)
    axins.set_frame_on(False)
    axins.set_xlim(snap.bh["pos"][bhi, 0]-R, snap.bh["pos"][bhi, 0]+R)
    axins.set_ylim(snap.bh["pos"][bhi, 2]-R, snap.bh["pos"][bhi, 2]+R)
    axins.set_aspect("equal")
    # add connecting lines manually
    for _ang in ang:
        point = convert_axes_coords_to_data_coords(
            ax=ax,
            coords=(
                bounds[0] + bounds[2]/2 * (1 + 0.95*np.cos(np.radians(_ang))), 
                bounds[1] + bounds[3]/2 * (1 + 0.95*np.sin(np.radians(_ang)))
            )
        )
        ax.plot([snap.bh["pos"][bhi,0], point[0]], [snap.bh["pos"][bhi,2], point[1]], lw=1, c="w")


fig, ax = plt.subplots(1, 2, figsize=(6, 3.5), sharex="all", sharey="all")
for i, (axi, s, ang) in enumerate(zip(ax, snapfiles, ((80, 240), (60, 180))),):
    snap = pygad.Snapshot(s, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)
    pygad.plotting.image(snap.stars, qty="mass", cmap="cmr.eclipse", showcbar=False, scaleind="line", ax=axi, extent=100, xaxis=0, yaxis=2)
    axi.set_facecolor("k")
    inset_plotter([0.78, 0.2, 0.2, 0.2], snap, axi, ang=ang)
    if i==0:
        inset_plotter([0.05, 0.65, 0.2, 0.2], snap, axi, bhi=1, ang=(250, 35))

bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "pr_plots/merger_overview.png"), force_ext=True, save_kwargs={"dpi":800})
