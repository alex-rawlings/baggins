import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
from tqdm import tqdm


snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")

xaxis = 0
yaxis = 1

use_2D = False

for i, snapfile in enumerate(snapfiles):
    if i != 5: continue
    snap = pygad.Snapshot(snapfile, physical=True)
    pygad.Translation(
        -pygad.analysis.shrinking_sphere(
            snap.stars, [0,0,0], 30
        )
    ).apply(snap, total=True)

    if use_2D:
        extent = pygad.utils.geo.dist(snap.bh["pos"])[0]
        print(f"BH is {extent:.1e} kpc away from centre of stellar mass")
        extent *= 1.3

        mask = pygad.ExprMask(f"abs(pos[:,{xaxis}]) < {np.abs(extent)}") & pygad.ExprMask(f"abs(pos[:,{yaxis}]) < {np.abs(extent)}")

        print("Plotting stellar mass distribution")
        fig, ax, im, cbar = pygad.plotting.image(snap.stars[mask], "mass", cbartitle="")
        ax.set_facecolor("k")
        ax.plot(snap.bh["pos"][:,0], snap.bh["pos"][:,1], marker="o", c="tab:red")
        ax.plot(0, 0, marker="x", c="tab:orange")

        print("Calculating contours")
        dens, xe, ye, bn = binned_statistic_2d(
                    x = snap.stars[mask]["pos"][:,0],
                    y = snap.stars[mask]["pos"][:,1], 
                    values = None,
                    statistic = "count",
                    bins = 10)
        xb = bgs.mathematics.get_histogram_bin_centres(xe)
        yb = bgs.mathematics.get_histogram_bin_centres(ye)
        CS = ax.contour(xb, yb, dens, 10, lw=2, cmap="Reds")
        #ax.clabel(CS, inline=True)
    else:
        r_edges = np.linspace(0.1, 30, 200)
        r_edges = np.insert(r_edges,0, [0])
        assert np.all(np.diff(r_edges) > 0)
        prof = pygad.analysis.profile_dens(snap.stars, "mass", r_edges=r_edges, proj=xaxis)
        plt.axvline(snap.bh["pos"][:,xaxis], c="k", alpha=0.5)
        plt.loglog(bgs.mathematics.get_histogram_bin_centres(r_edges), prof, lw=2)
        
    plt.show()
    quit()