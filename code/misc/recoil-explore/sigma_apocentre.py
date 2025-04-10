import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygad
import baggins as bgs

bgs.plotting.check_backend()

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_007.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

# projected dispersion for bound particles
bound_ids = bgs.analysis.find_individual_bound_particles(snap)
id_mask = pygad.IDMask(bound_ids)

pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
rinfl = list(bgs.analysis.influence_radius(snap).values())[0]

print(f"rinfl: {rinfl}")

radii = np.geomspace(1e-2, rinfl, 10)
sigmas = np.full_like(radii, np.nan)

for i, r in enumerate(radii):
    mask = pygad.BallMask(r) & id_mask
    sigmas[i] = pygad.analysis.los_velocity_dispersion(snap.stars[mask], proj = 1)

plt.loglog(radii, sigmas, "-o")
plt.xlabel("r/kpc")
plt.ylabel("sigma/(km/s)")
plt.savefig("sigma_with_r.png", dpi=300)

plt.close()

# IFU map, but in a slice ~ in the plane of the BH
# set up the instruments
fig, ax = plt.subplots(1, 2)
muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = 0.6
seeing = {"num": 25, "sigma": muse_nfm.resolution_kpc}
ifu_mask = pygad.ExprMask(
    f"abs(pos[:,{0}]) <= {0.5 * muse_nfm.extent}"
) & pygad.ExprMask(
    f"abs(pos[:,{2}]) <= {0.5 * muse_nfm.extent}"
) & pygad.ExprMask(
    f"abs(pos[:,{1}]) <= {5}"
)
bgs.analysis.basic_snapshot_centring(snap)
print(f"BH at: {snap.bh['pos']}")

'''inject_mask = pygad.BallMask(1, [5, 0, 5])
ids = snap.stars[inject_mask]["ID"]
id_mask = np.zeros(len(snap.stars), dtype=bool)
for _id in tqdm(ids, total=len(ids)):
    id_mask[snap.stars["ID"]==_id] = True
snap.stars["vel"][id_mask,1] = 1e5
print(f"Max stellar vel: {np.max(snap.stars['vel'])}")'''

voronoi = bgs.analysis.VoronoiKinematics(
    x=snap.stars[ifu_mask]["pos"][:, 0],
    y=snap.stars[ifu_mask]["pos"][:, 2],
    V=snap.stars[ifu_mask]["vel"][:, 1],
    m=snap.stars[ifu_mask]["mass"],
    Npx=muse_nfm.number_pixels,
    seeing=seeing,
)
voronoi.make_grid(part_per_bin=int(600**2))
voronoi.binned_LOSV_statistics()
voronoi.plot_kinematic_maps(moments="12", ax=ax, cbar="inset")
for axi in ax:
    axi.scatter(snap.bh["pos"][:,0], snap.bh["pos"][:,2], c="k")
    axi.set_xlabel("x/kpc")
    axi.set_ylabel("z/kpc")
plt.savefig("sigma_ifu_slice.png", dpi=300)
plt.close()

