import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.check_backend()

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0300/output/snap_003.hdf5"
snap = pygad.Snapshot(snapfile, physical=True)

if False:
    bound_ids, frac, bind_energies = bgs.analysis.find_individual_bound_particles(snap, return_extra=True)

    thresh = 0
    energy_normed = -bind_energies[bind_energies < 0]/snap.stars["mass"][0]
    print(f"The mass is {np.sum(energy_normed>thresh)*snap.stars['mass'][0]:.1e} strongly bound stars")

    pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
    pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

    id_mask = pygad.IDMask(bound_ids[energy_normed>thresh])
    vsig = pygad.analysis.los_velocity_dispersion(snap.stars[id_mask], proj=1)
    print(f"Corresponding LOS vel dispersion is {vsig:.2e}")
    reff = pygad.analysis.half_mass_radius(snap.stars[id_mask], proj=1, center=snap.bh["pos"].flatten())
    print(f"Corresponding Reff is {reff:.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(7,4))
    ax[0].hist(energy_normed, 20)
    ax[0].axvline(thresh, c="tab:red")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Normalised binding energy")
    ax[0].set_ylabel("Count")

    idx = np.argsort(id_mask._IDs)

    ax[1].plot(snap.stars[id_mask]["r"][idx], energy_normed[idx], ls="", marker=".")
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("Normalised binding energy")

    ax[2].hist(snap.stars[id_mask]["vel"][:,1], bins=20)
    ax[2].set_xlabel("LOS velocity")
    ax[2].set_ylabel("Count")

    print("Showing")
    plt.show()

if False:
    G = pygad.UnitScalar(4.3009e-6, "kpc/Msol*(km/s)**2")
    # from Merrit 2009
    pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
    pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

    bound_mask = snap.stars["vel"][:,0]**2 + snap.stars["vel"][:,1]**2 + snap.stars["vel"][:,2]**2 < 2 * G  * snap.bh["mass"].flatten() / snap.stars["r"]
    id_mask = pygad.IDMask(snap.stars[bound_mask]["ID"])

    mass_bound = np.sum(snap.stars[id_mask]["mass"])
    print(f"Bound mass {mass_bound:.1e}")
    vsig = pygad.analysis.los_velocity_dispersion(snap.stars[id_mask], proj=1)
    print(f"Corresponding LOS vel dispersion is {vsig:.2e}")
    reff = pygad.analysis.half_mass_radius(snap.stars[id_mask], proj=1, center=snap.bh["pos"].flatten())
    print(f"Corresponding Reff is {reff:.2e}")

    fig, ax = plt.subplots(1, 2, sharex="all")
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].plot(snap.stars[id_mask]["pos"][:,0], snap.stars[id_mask]["pos"][:,1], ls="", marker=".")
    ax[1].plot(snap.stars[id_mask]["pos"][:,0], snap.stars[id_mask]["pos"][:,2], ls="", marker=".")

    plt.show()

if True:
    bgs.analysis.basic_snapshot_centring(snap)
    cluster_data = bgs.utils.load_data("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/perfect_obs/perf_obs_0600.pickle")
    snapnum = bgs.general.get_snapshot_number("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_007.hdf5")
    rmax = cluster_data["cluster_props"][int(snapnum)-2]["r_centres_cluster"][-1] - snap.bh["r"].flatten()
    print(rmax)
    pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
    mask1 = pygad.BallMask(rmax[0], center=snap.bh["pos"].flatten())
    mask2 = bgs.analysis.get_cylindrical_mask(rmax[0], proj=1)
    for mask in (mask1, mask2):
        v = snap.stars[mask]['vel'][:,1]
        sig = np.nanstd(v)
        print(f"Disp: {sig:.4e}")
        v = v[np.abs(v) < 3 * sig]
        sig = np.nanstd(v)
        print(f"Disp: {sig:.4e}")
        plt.hist(v, 20, alpha=0.5, density=True)
    plt.savefig("disp.png", dpi=300)