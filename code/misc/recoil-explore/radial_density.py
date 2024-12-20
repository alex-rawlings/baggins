import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad

bgs.plotting.check_backend()

snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")

fig, ax = plt.subplots(1, 3, sharex="all", sharey="all")

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
    # plot density of all particles
    Rmax = list(bgs.analysis.influence_radius(snap, combined=True).values())[0]
    pygad.plotting.profile(snap.stars, Rmax=Rmax, ax=ax[i], label="all", **plot_kwargs)
    pygad.plotting.profile(snap.stars[bound_mask], Rmax=Rmax, ax=ax[i], label="bound only", **plot_kwargs)
ax[0].legend()
ax[0].set_ylim(1e5, 5e10)
ax[0].set_title("Before kick")
ax[1].set_title("1st Apocentre")
ax[2].set_title("1st Pericentre")
bgs.plotting.savefig("radial_density.png")

