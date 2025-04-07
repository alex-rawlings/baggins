import os.path
import matplotlib.pyplot as plt
import pygad
from PIL import Image
import baggins as bgs


bgs.plotting.check_backend()

x_axis = 0
y_axis = 2
LOS_axis = 1

muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = 0.6
euclid_vis = bgs.analysis.Euclid_VIS()
euclid_vis.redshift = muse_nfm.redshift

print(f"Euclid pixel width: {euclid_vis.pixel_width}")
print(f"Muse pixel width: {muse_nfm.pixel_width}")

ifu_mask = pygad.ExprMask(
    f"abs(pos[:,{x_axis}]) <= {0.5 * muse_nfm.extent}"
) & pygad.ExprMask(f"abs(pos[:,2]) <= {0.5 * muse_nfm.extent}")
seeing = {"num": 25, "sigma": muse_nfm.pixel_width}

snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")

animation_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/misc/voronoi_sigma_only"

contour_kwargs = {"linestyles":"solid", "linewidths":0.5, "levels":10, "colors":"k"}

if True:
    for i, snapfile in enumerate(snapfiles):
        if i > 44:
            break
        print(f"Doing snapshot {i:03d}")
        fig, ax = plt.subplots(1, 3)
        fig.set_figwidth(2*fig.get_figwidth())

        snap = pygad.Snapshot(snapfile, physical=True)
        bgs.analysis.basic_snapshot_centring(snap)

        fig.suptitle(f"Time: {bgs.general.convert_gadget_time(snap):.2f} Gyr")

        # plot the density and save
        _, _, im = pygad.plotting.image(
            snap.stars[ifu_mask],
            qty="mass",
            xaxis=x_axis,
            yaxis=y_axis,
            scaleind="labels",
            ax=ax[0],
            vlim=(1e7, 5e9),
            Npx=euclid_vis.number_pixels,
            showcbar=False
        )
        
        ax[0].contour(im.get_array(), extent=im.get_extent(), **contour_kwargs)

        voronoi = bgs.analysis.VoronoiKinematics(
            x=snap.stars[ifu_mask]["pos"][:, x_axis],
            y=snap.stars[ifu_mask]["pos"][:, 2],
            V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
            m=snap.stars[ifu_mask]["mass"],
            Npx=muse_nfm.number_pixels,
            seeing=seeing
        )
        voronoi.make_grid(part_per_bin=5000 * seeing["num"])
        voronoi.binned_LOSV_statistics(p=2)

        # TODO update the voronoi plotting method, this is a hack for now
        voronoi.plot_kinematic_maps(ax=ax[1:], clims={"V":10, "sigma":[170, 270]}, cbar="inset")
        for axi in ax[1:]:
            axi.set_xlabel("x/kpc")
            axi.set_ylabel("z/kpc")

        bgs.plotting.savefig(os.path.join(animation_path, f"frame_{i:03d}.png"))

        # save memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()


frames = bgs.utils.get_files_in_dir(animation_path, ".png")

frames = [Image.open(img) for img in frames if "frame_" in img]  # Load images
frames.extend([frames[-1]] * 10)

# Save as an animated GIF
frames[0].save(
    os.path.join(animation_path, f"animation_0600.gif"),
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=200,
    loop=0,
)

plt.close()