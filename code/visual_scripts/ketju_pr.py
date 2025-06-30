import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pygad
import baggins as bgs


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/trail_blazer/H_2M_b-H_2M_c-30.000-2.000/output/snap_006.hdf5"
R = 0.7
rng = np.random.default_rng(54533)
num_points = 300


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

    points = rng.normal(-1, 1, size=(num_points, 2))

    # Find the center-most point (closest to the centroid)
    centroid = points.mean(axis=0)
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    base_index = np.argmin(distances_to_centroid)
    base_point = points[base_index]
    points -= base_point

    # Identify points within the circle
    distances_to_base = np.linalg.norm(points - base_point, axis=1)
    points = points[np.where(distances_to_base <= 1.2 * R)[0]]
    distances_to_base = np.linalg.norm(points - base_point, axis=1)
    in_circle_indices = np.where(distances_to_base <= R)[0]
    in_circle_points = points[in_circle_indices]

    # Recompute MST only using points inside the circle
    sub_dist_matrix = distance_matrix(in_circle_points, in_circle_points)
    sub_mst = minimum_spanning_tree(sub_dist_matrix).toarray()

    # Plot
    axins.scatter(
        points[:, 0],
        points[:, 1],
        c="goldenrod",
        marker="*",
        ec="w",
        lw=0.05,
        s=10,
        zorder=0.5,
    )
    axins.scatter(*base_point, color="black", s=20, ec="w", lw=0.5)

    # Draw MST edges among in-circle points
    for i in range(len(in_circle_points)):
        for j in range(len(in_circle_points)):
            if sub_mst[i, j] > 0:
                axins.plot(
                    [in_circle_points[i, 0], in_circle_points[j, 0]],
                    [in_circle_points[i, 1], in_circle_points[j, 1]],
                    "w-",
                    zorder=0.3,
                    lw=0.3,
                )

    # Draw the circle
    circle = Circle(
        base_point,
        R,
        edgecolor="w",
        facecolor="none",
        linestyle=":",
        linewidth=0.8,
        zorder=0.2,
    )
    axins.add_patch(circle)
    axins.set_facecolor("k")
    circ = Circle(
        (0.5, 0.5),
        0.48,
        transform=axins.transAxes,
        facecolor="k",
        edgecolor="w",
        zorder=0.01,
    )
    axins.set_clip_path(circ)
    axins.add_patch(circ)
    axins.set_frame_on(False)
    axins.set_xlim(-1, 1)
    axins.set_ylim(-1, 1)
    axins.set_aspect("equal")
    # add connecting lines manually
    for _ang in ang:
        point = convert_axes_coords_to_data_coords(
            ax=ax,
            coords=(
                bounds[0] + bounds[2] / 2 * (1 + 0.95 * np.cos(np.radians(_ang))),
                bounds[1] + bounds[3] / 2 * (1 + 0.95 * np.sin(np.radians(_ang))),
            ),
        )
        ax.plot(
            [snap.bh["pos"][bhi, 0], point[0]],
            [snap.bh["pos"][bhi, 2], point[1]],
            lw=1,
            c="w",
        )


fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
snap = pygad.Snapshot(snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)
pygad.plotting.image(
    snap.stars,
    qty="mass",
    cmap="cmr.eclipse",
    showcbar=False,
    scaleind="line",
    ax=ax,
    extent=100,
    xaxis=0,
    yaxis=2,
)
ax.set_facecolor("k")
inset_plotter([0.58, 0.2, 0.4, 0.4], snap, ax, ang=(95, 210))
inset_plotter([0.05, 0.55, 0.4, 0.4], snap, ax, bhi=1, ang=(250, 0))

bgs.plotting.savefig(
    os.path.join(bgs.FIGDIR, "pr_plots/ketju.png"),
    force_ext=True,
    save_kwargs={"dpi": 800},
)
