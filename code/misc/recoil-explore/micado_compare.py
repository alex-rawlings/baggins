import os.path
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.check_backend()

# let's look at this case:
kv = 600
snapfile = bgs.utils.get_snapshots_in_dir(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv:04d}/output"
)[7]

# set up the instruments
muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = 0.6
muse_seeing = {"num": 25, "sigma": muse_nfm.resolution_kpc}

micado = bgs.analysis.MICADO_NFM()
micado.redshift = 0.6
micado_seeing = {"num": 25, "sigma": micado.resolution_kpc}



snap = pygad.Snapshot(snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)
snapnum = bgs.general.get_snapshot_number(snapfile)
print(f"Doing snapshot {snapnum}")
fig, ax = plt.subplots(1, 2, sharex="all", sharey="all") 
fig.set_figwidth(2*fig.get_figwidth())

for i, (axi, instrument, seeing)in enumerate(zip(ax, (muse_nfm, micado), (muse_seeing, micado_seeing))):
    print(f"Doing {instrument.name}")
    ifu_mask = pygad.ExprMask(
        f"abs(pos[:,0]) <= {0.5 * instrument.extent}"
    ) & pygad.ExprMask(f"abs(pos[:,2]) <= {0.5 * instrument.extent}")
    # voronoi
    voronoi = bgs.analysis.VoronoiKinematics(
        x=snap.stars[ifu_mask]["pos"][:, 0],
        y=snap.stars[ifu_mask]["pos"][:, 2],
        V=snap.stars[ifu_mask]["vel"][:, 1],
        m=snap.stars[ifu_mask]["mass"],
        Npx=instrument.number_pixels,
        seeing=seeing,
    )
    voronoi.make_grid(part_per_bin=int((10**(3+i*2))**2))
    voronoi.binned_LOSV_statistics()
    voronoi.plot_kinematic_maps(ax=axi, moments="2", cbar="inset")
    axi.set_xlabel("x/kpc")
    axi.set_ylabel("z/kpc")
    axi.set_title(instrument.name)
    axi.scatter(snap.bh["pos"][:,0], snap.bh["pos"][:,2], marker="o", fc="none", ec="k", lw=0.3, s=30)

bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"kicksurvey-study/muse_micado.png"))
