import numpy as np
import matplotlib.pyplot as plt
import pygad
from tqdm import tqdm
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

if False:
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

if True:
    bgs.analysis.basic_snapshot_centring(snap)
    fig, ax = plt.subplots(1, 3, figsize=(7, 3))

    rinfl = list(bgs.analysis.influence_radius(snap).values())[0]
    bound_ids, _, energy = bgs.analysis.find_individual_bound_particles(snap, return_extra=True)
    ball_mask = pygad.BallMask(3*rinfl, center=snap.bh["pos"].flatten())
    vsig = np.nanstd(snap.stars[ball_mask]["vel"])
    strong_bound_ids = bound_ids[energy[energy<0]/vsig**2 < -1]
    bound_id_mask = pygad.IDMask(strong_bound_ids)
    energy = energy[energy < 0] / vsig**2

    # Look at dispersion comparison
    #rinfl_cylinder_mask = bgs.analysis.get_cylindrical_mask(rinfl, proj=1, centre=snap.bh["pos"].flatten()[[0,2]])
    bins = np.arange(-1000, 1001, 100)
    def get_sigma(v):
        return np.linalg.norm(np.nanstd(v, axis=0))
    ax[0].hist(snap.stars[ball_mask]["vel"].flatten(), bins=bins, label=f"ambient: sig={get_sigma(snap.stars[ball_mask]['vel']):.2e}")
    ax[0].hist(snap.stars[bound_id_mask]["vel"].flatten(), bins=bins, label=f"Strongly bound: sig={get_sigma(snap.stars[bound_id_mask]['vel']):.2e}")
    '''
    v = snap.stars[rinfl_cylinder_mask]["vel"][:, 1]
    ax[0].hist(v, bins=bins, alpha=0.6, label=f"all, sig={np.nanstd(v):.2e}")
    v = v[np.abs(v) < 3 * np.nanstd(v)]
    ax[0].hist(v, bins=bins, alpha=0.6, label=f"cut, sig={np.nanstd(v):.2e}")'''
    ax[0].legend(fontsize="x-small")
    ax[0].set_xlabel("3D velocity")
    ax[0].set_ylabel("Count")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, ax[0].get_ylim()[1])

    ax[1].hist(energy)#, bins=np.arange(-10, 0.1, 1))
    ax[1].set_xlabel("Binding energy (<rinfl)")
    ax[1].set_ylabel("Count")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, ax[1].get_ylim()[1])
    for axi in ax[1:]:
        axi.axvline(-1, c="tab:red", label="ambient sigma")

    sorted_idx = np.argsort(energy)
    stride = 100
    ids = []
    reff = []
    for i, idx in tqdm(enumerate(sorted_idx), total=len(sorted_idx), desc="Eff. rad calculation"):
        ids.append(bound_ids[idx])
        if i%stride != 0:
            continue
        id_mask = pygad.IDMask(ids)
        reff.append(pygad.analysis.half_mass_radius(snap.stars[id_mask], center=snap.bh["pos"].flatten(), proj=1))
    ax[2].plot(energy[sorted_idx][::stride], reff)
    ax[2].set_xlabel("Energy")
    ax[2].set_ylabel("Calculated effective radius")
    ax[2].grid(which="major")
    bgs.plotting.savefig("disp_and_energy.png")
