import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad


# let's see where the stars that are within the influence radius of the BH end up
snapdir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output"

snapnums = [0, 10, 37, 38, 39, 40, 41, 42, 43, 44, 45, 73]
xaxis = 0
yaxis = 1
dt = pygad.UnitScalar("0.1 Myr")

snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)

fig, ax = plt.subplots(3,4, sharex=True, sharey=True)

for i, (axi, sn) in enumerate(zip(ax.flat, snapnums)):
    print(f"Doing snapshot number {sn}")
    snap = pygad.Snapshot(snapfiles[sn], physical=True)

    # centre it
    xcom = pygad.analysis.shrinking_sphere(
        snap.stars,
        pygad.analysis.center_of_mass(snap.stars),
        30
    )
    pygad.Translation(-xcom).apply(snap, total=True)

    # find those particles within the influence radius at snapshot 0
    if i==0:
        rinfl = list(bgs.analysis.influence_radius(snap, combined=True).values())[0]
        print(f"  rinfl is {rinfl:.2f}")
        rinfl_mask = pygad.BallMask(rinfl)
        id_mask = pygad.IDMask(snap.stars[rinfl_mask]["ID"])
    mask = pygad.masks.ExprMask("abs(pos[:,2])<0.5") & id_mask
    pygad.plotting.image(snap.stars[mask], "mass", xaxis=xaxis, yaxis=yaxis, zero_is_white=False, ax=axi, vlim=(1e7, 1e9))
    axi.set_title(f"Snap {sn:03d}")
    axi.scatter(snap.bh["pos"][:,xaxis], snap.bh["pos"][:,yaxis], color="tab:red", marker="o")
    if i != 0:
        axi.arrow(
            snap.bh["pos"][0,xaxis], snap.bh["pos"][0,yaxis],
            snap.bh["vel"][0,xaxis]*dt, snap.bh["vel"][0,yaxis]*dt,
            color="tab:red",
            length_includes_head = True,
            head_width = 0.1
        )

    axi.set_aspect("equal")
    axi.set_facecolor("k")


    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

plt.show()