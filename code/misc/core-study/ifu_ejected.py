import matplotlib.pyplot as plt
import baggins as bgs
import pygad


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_020.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)
assert len(snap.bh) == 1
#rhalf = pygad.analysis.half_mass_radius(snap.stars)
# TODO why is this not changing with frac??
rhalf = list(
    bgs.analysis.enclosed_mass_radius(snap, 2).values()
)[0]
print(rhalf)
ball_mask = pygad.BallMask(rhalf, snap.bh["pos"][0,:])
print(snap.stars[~ball_mask]["pos"][:2,:])
pygad.analysis.orientate_at(snap[ball_mask], "red I", total=True)
print(snap.stars[~ball_mask]["pos"][:2,:])
quit()


voronoi_stats = bgs.analysis.voronoi_binned_los_V_statistics(
    snap.stars[ball_mask]["pos"][:, 0],
    snap.stars[ball_mask]["pos"][:, 1],
    snap.stars[ball_mask]["vel"][:, 2],
    snap.stars[ball_mask]["mass"],
    part_per_bin=2000
)

ax = bgs.plotting.voronoi_plot(voronoi_stats)

for axi in ax:
    axi.plot(snap.bh["pos"][0,0], snap.bh["pos"][0,1], marker="o", c="k")
fig = axi.get_figure()
fig.suptitle(f"t={bgs.general.convert_gadget_time(snap):.2f} Gyr, r_BH={bgs.mathematics.radial_separation(snap.bh['pos'])[0]:.2f} kpc")

bgs.plotting.savefig("v0600-20-ifu.png")
plt.show()