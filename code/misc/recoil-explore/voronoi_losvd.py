import matplotlib.pyplot as plt
import pygad
import h5py
import baggins as bgs

bgs.plotting.check_backend()

x_axis = 0
y_axis = 2
LOS_axis = 1

muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = 1.

ifu_mask = pygad.ExprMask(
    f"abs(pos[:,{x_axis}]) <= {0.5 * muse_nfm.extent}"
) & pygad.ExprMask(f"abs(pos[:,2]) <= {0.5 * muse_nfm.extent}")
seeing = {"num": 25, "sigma": muse_nfm.pixel_width}

fig, ax = plt.subplots(2, 3, sharex="all", sharey="all")
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(1.5 * fig.get_figheight())

data_bh = []
data_other = []

for i, snapnum, in enumerate((0, 9, 16)):
    snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_{snapnum:03d}.hdf5"
    print(f"Doing snapshot {snapnum:03d}")

    snap = pygad.Snapshot(snapfile, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)

    voronoi = bgs.analysis.VoronoiKinematics(
        x=snap.stars[ifu_mask]["pos"][:, x_axis],
        y=snap.stars[ifu_mask]["pos"][:, 2],
        V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
        m=snap.stars[ifu_mask]["mass"],
        Npx=muse_nfm.number_pixels,
        seeing=seeing
    )
    voronoi.make_grid(part_per_bin=5000 * seeing["num"])
    voronoi.binned_LOSV_statistics(p=6)

    # plot for BH
    data_bh.append(voronoi.get_pixel_LOSVD(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis])[0])
    voronoi.plot_pixel_LOSVD(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis], bins=50, ax=ax[0,i])
    ax[0,i].text(0.1, 0.9, "BH", transform=ax[0,i].transAxes)
    ax[0,i].set_title(f"Snapshot {snapnum:03d}")

    # plot for other
    data_other.append(voronoi.get_pixel_LOSVD(-10, 0)[0])
    voronoi.plot_pixel_LOSVD(-10, 0, bins=50, ax=ax[1,i])
    ax[1,i].text(0.1, 0.9, "Opposite peak", transform=ax[1,i].transAxes)
ax[0,0].set_xlim(-1e3, 1e3)
bgs.plotting.savefig("losvd.png")

if False:
    with h5py.File("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/misc/losvd_jens.hdf5", "w") as f:
        f.create_dataset("BH_merger", data=data_bh[0])
        f["BH_merger"].attrs["description"] = "LOSVD of pixel containing BH at time of merger"
        f.create_dataset("BH_apo", data=data_bh[1])
        f["BH_apo"].attrs["description"] = "LOSVD of pixel containing BH at time of first apocentre"
        f.create_dataset("BH_peri", data=data_bh[2])
        f["BH_peri"].attrs["description"] = "LOSVD of pixel containing BH at time of first pericentre"
        f.create_dataset("peak_merger", data=data_other[0])
        f["peak_merger"].attrs["description"] = "LOSVD of pixel at coordinates (-10, 0) at time of BH merger"
        f.create_dataset("peak_apo", data=data_other[1])
        f["peak_apo"].attrs["description"] = "LOSVD of pixel at coordinates (-10, 0) at time of BH apocentre"
        f.create_dataset("peak_peri", data=data_other[2])
        f["peak_peri"].attrs["description"] = "LOSVD of pixel at coordinates (-10, 0) at time of BH pericentre"
