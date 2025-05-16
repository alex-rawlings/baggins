import matplotlib.pyplot as plt
import pygad
import os.path
import h5py
import baggins as bgs

bgs.plotting.check_backend()

x_axis = 1
y_axis = 2
LOS_axis = list(set({0,1,2}).difference({x_axis,y_axis}))[0]
if LOS_axis == 0:
    moment = "1"
else:
    moment = "2"

harmoni = bgs.analysis.HARMONI_SPATIAL(z=0.6)
harmoni.max_extent = 5
seeing = {"num": 25, "sigma": harmoni.pixel_width.value}
ifu_mask = harmoni.get_fov_mask(x_axis, y_axis)
fit_order = 16

'''fig, ax = plt.subplots(2, 3, sharex="all", sharey="all")
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(1.5 * fig.get_figheight())'''

fig, ax = plt.subplots(1, 2)
fig.set_figwidth(2 * fig.get_figwidth())

data_bh = []
data_other = []

if False:
    for i, snapnum, in enumerate((4,)):
        snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_{snapnum:03d}.hdf5"
        print(f"Doing snapshot {snapnum:03d}")

        snap = pygad.Snapshot(snapfile, physical=True)
        bgs.analysis.basic_snapshot_centring(snap)

        voronoi = bgs.analysis.VoronoiKinematics(
            x=snap.stars[ifu_mask]["pos"][:, x_axis],
            y=snap.stars[ifu_mask]["pos"][:, y_axis],
            V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
            m=snap.stars[ifu_mask]["mass"],
            Npx=harmoni.number_pixels,
            seeing=seeing
        )
        voronoi.make_grid(part_per_bin=int(400**2))
        voronoi.binned_LOSV_statistics(p=fit_order)
        bgs.utils.save_data(voronoi.dump_to_dict(), f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/ifu_high_order/harmoni_fit_{fit_order}_snap_{bgs.general.get_snapshot_number(snapfile)}.pickle", exist_ok=True)

        '''# plot the voronoi bins
        voronoi.plot_kinematic_maps(ax=ax[0], cbar="inset")

        # plot LOSVD at BH pixel
        voronoi.plot_pixel_LOSVD(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis], bins=50, ax=ax[1])

        ax[0].scatter(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis], fc="none", ec="k", lw=0.5, s=100)

        if True:
            ax[0].scatter(
                snap.bh["pos"][:, x_axis],
                snap.bh["pos"][:, y_axis],
                fc="none",
                lw=0.5,
                ec="k",
                s=100
            )'''

        '''# plot for BH
        data_bh.append(voronoi.get_pixel_LOSVD(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis])[0])
        voronoi.plot_pixel_LOSVD(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis], bins=50, ax=ax[0,i])
        ax[0,i].text(0.1, 0.9, "BH", transform=ax[0,i].transAxes)
        ax[0,i].set_title(f"Snapshot {snapnum:03d}")

        # plot for other
        data_other.append(voronoi.get_pixel_LOSVD(-10, 0)[0])
        voronoi.plot_pixel_LOSVD(-10, 0, bins=50, ax=ax[1,i])
        ax[1,i].text(0.1, 0.9, "Opposite peak", transform=ax[1,i].transAxes)'''
    #ax[0,0].set_xlim(-1e3, 1e3)
    #bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/losvd.png"))

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

if True:
    data_file = bgs.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/ifu_high_order", ".pickle")[0]
    print(data_file)
    assert "004" in data_file
    data = bgs.utils.load_data(data_file)
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(data)

    snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_004.hdf5"

    snap = pygad.Snapshot(snapfile, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)

    fig, ax = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(6, 6))
    voronoi.plot_kinematic_maps(ax=ax, cbar="inset")
    for axi in ax.flat:
        axi.scatter(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis], fc="none", ec="k", lw=0.5, s=100)
    bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/losvd_high.png"))
    plt.close()

    voronoi.plot_pixel_LOSVD(snap.bh["pos"][0,x_axis], snap.bh["pos"][0,y_axis], bins=30)
    bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/losvd_high_bh_pixel.png"))


