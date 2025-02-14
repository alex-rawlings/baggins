import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad

bgs.plotting.check_backend()

snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")

fig, ax = plt.subplots(2, 3, sharex="all", sharey="row")

plot_kwargs = dict(
    qty = "mass",
    N = 20,
    logbin=True,
    lw = 2,
)

for i, snapnum in enumerate((0, 9, 16)):
    print(f"Doing snapshot {snapnum:03d}")
    snap = pygad.Snapshot(snapfiles[snapnum], physical=True)
    xcom = pygad.analysis.center_of_mass(snap.bh)
    pygad.Translation(-xcom).apply(snap, total=True)
    # get bound particles
    bound_stars = bgs.analysis.find_individual_bound_particles(snap)
    if len(bound_stars) == 0:
        print(f"No bound stars for snapshot {snapnum:03d}")
    bound_mask = pygad.IDMask(bound_stars)
    # plot 3D density
    Rmax = list(bgs.analysis.influence_radius(snap, combined=True).values())[0]
    # of bound particles
    pygad.plotting.profile(snap.stars[bound_mask], Rmax=Rmax, ax=ax[0,i], label="bound only", **plot_kwargs)
    # of all particles within influence radius
    pygad.plotting.profile(snap.stars, Rmax=Rmax, ax=ax[0,i], label="stars < rinfl", **plot_kwargs)
    print(f" -> Half mass radius: {pygad.analysis.half_mass_radius(snap.stars[bound_mask])}")
    print(f" -> LOS vel disp.: {pygad.analysis.los_velocity_dispersion(snap.stars[bound_mask], proj=1)}")

    # plot projected density
    # of bound particles
    pygad.plotting.profile(snap.stars[bound_mask], Rmax=Rmax, proj=1, ax=ax[1,i], label="bound cluster", **plot_kwargs)
    # of entire galaxy
    pygad.Translation(xcom).apply(snap, total=True)
    bgs.analysis.basic_snapshot_centring(snap)
    pygad.plotting.profile(snap.stars, Rmax=Rmax, proj=1, ax=ax[1,i], label="galaxy", **plot_kwargs)
    ax[1,i].axvline(snap.bh["r"][0], c="k", lw=1)
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

for i in range(2):
    ax[i,0].legend()
ax[0,0].set_ylim(1e5, 5e10)
ax[0,0].set_title("Before kick")
ax[0,1].set_title("1st Apocentre")
ax[0,2].set_title("1st Pericentre")
bgs.plotting.savefig("radial_density.png")

