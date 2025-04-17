import os.path
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.check_backend()

# let's look at this case:
kv = 600
snapfiles = bgs.utils.get_snapshots_in_dir(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv:04d}/output"
)

# set up the instruments
muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = 0.6
seeing = {"num": 25, "sigma": muse_nfm.resolution_kpc}

first_done = False
for snapfile in snapfiles[:8]:
    # load and centre snap
    snap = pygad.Snapshot(snapfile, physical=True)
    if len(snap.bh) > 1:
        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
        continue
    bgs.analysis.basic_snapshot_centring(snap)
    snapnum = bgs.general.get_snapshot_number(snapfile)
    print(f"Doing snapshot {snapnum}")
    fig, ax = plt.subplots(2, 2, sharex="col", sharey="col")
    ax[-1, -1].set_xlabel("LOSVD [km/s]")
    fig.set_figwidth(2*fig.get_figwidth())
    fig.set_figheight(1.5 * fig.get_figheight())
    fig.suptitle(f"Snapshot: {snapnum}")
    for i, LOS in enumerate((1, 0)):
        xaxis = list(set({0, 1, 2}).difference({LOS}))[0]
        if LOS == 0:
            moment = "1"
        else:
            moment = "2"
        ifu_mask = pygad.ExprMask(
            f"abs(pos[:,{xaxis}]) <= {0.5 * muse_nfm.extent}"
        ) & pygad.ExprMask(f"abs(pos[:,2]) <= {0.5 * muse_nfm.extent}")
        # voronoi
        voronoi = bgs.analysis.VoronoiKinematics(
            x=snap.stars[ifu_mask]["pos"][:, xaxis],
            y=snap.stars[ifu_mask]["pos"][:, 2],
            V=snap.stars[ifu_mask]["vel"][:, LOS],
            m=snap.stars[ifu_mask]["mass"],
            Npx=muse_nfm.number_pixels,
            seeing=seeing,
        )
        voronoi.make_grid(part_per_bin=int(1000**2))
        voronoi.binned_LOSV_statistics()
        voronoi.plot_kinematic_maps(ax=ax[i,0], moments=moment, cbar="inset")
        ax[i,0].set_xlabel(f"{'xy'[xaxis]}/kpc")
        ax[i,0].set_ylabel("z/kpc")
        ax[i,0].scatter(snap.bh["pos"][:,xaxis], snap.bh["pos"][:,2], marker="o", fc="k", ec="w", lw=0.3)
        voronoi.plot_pixel_LOSVD(snap.bh["pos"][0,xaxis], snap.bh["pos"][0,2], ax=ax[i,1], density=False, bins=50, ec="k", lw=0.2)
        ax[i,1].set_yscale("log")
        ax[i,1].set_ylabel("Counts")
    ax[1,1].set_ylim(1, ax[1,1].get_ylim()[1])
    if not first_done:
        losvd_xlim = ax[1,1].get_xlim()
        first_done = True
    else:
        ax[1,1].set_xlim(*losvd_xlim)
    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()
    bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"kicksurvey-study/LOSVD_series/vk_{kv:04d}_snap_{snapnum}.png"))
