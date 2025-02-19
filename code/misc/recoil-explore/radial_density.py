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

    # centre on SMBH
    xcom = pygad.analysis.center_of_mass(snap.bh)
    pygad.Translation(-xcom).apply(snap, total=True)
    # get bound particles
    bound_star_ids = bgs.analysis.find_individual_bound_particles(snap)
    if len(bound_star_ids) == 0:
        print(f"No bound stars for snapshot {snapnum:03d}")
    bound_mask = pygad.IDMask(bound_star_ids)

    # plot 3D density
    Rinfl = list(bgs.analysis.influence_radius(snap, combined=True).values())[0]
    # of bound particles within influence radius
    pygad.plotting.profile(snap.stars[bound_mask], Rmax=Rinfl, ax=ax[0,i], label="bound only", **plot_kwargs)
    # of all particles within influence radius
    pygad.plotting.profile(snap.stars, Rmax=Rinfl, ax=ax[0,i], label="stars < rinfl", **plot_kwargs)
    print(f" -> Half mass radius: {pygad.analysis.half_mass_radius(snap.stars[bound_mask]):.2f} kpc")
    print(f" -> Influence radius: {Rinfl:.2f} kpc")
    print(f" -> Furthest bound particle is {np.max(snap.stars[bound_mask]['r']):.2f} kpc from the BH")
    print(f" -> LOS vel disp.: {pygad.analysis.los_velocity_dispersion(snap.stars[bound_mask], proj=1):.2f} km/s")

    # plot projected density
    # of bound particles
    cluster_redges = np.geomspace(1e-2, Rinfl, 11)
    bound_proj = pygad.analysis.profile_dens(snap.stars[bound_mask], qty="mass", r_edges=cluster_redges, proj=1)
    bgs.analysis.basic_snapshot_centring(snap)
    ax[1,i].plot(bgs.mathematics.get_histogram_bin_centres(cluster_redges)+snap.bh["r"][0], bound_proj, label="bound cluster", lw=2)
    Rmax = 30
    # of entire galaxy
    pygad.plotting.profile(snap.stars, Rmax=Rmax, proj=1, ax=ax[1,i], label="galaxy", **plot_kwargs)
    #ax[1,i].axvline(snap.bh["r"][0], c="k", lw=1)
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

for i in range(2):
    ax[i,0].legend()
    ax[0,i+1].set_ylabel("")
    ax[1,i+1].set_ylabel("")
for i in range(3):
    ax[0,i].set_xlabel("")

ax[0,0].set_ylim(1e5, 5e10)
ax[0,0].set_title("Before kick")
ax[0,1].set_title("1st Apocentre")
ax[0,2].set_title("1st Pericentre")
bgs.plotting.savefig("radial_density.png")

