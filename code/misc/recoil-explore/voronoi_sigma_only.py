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
muse_nfm.redshift = 0.3

ifu_mask = pygad.ExprMask(
    f"abs(pos[:,{x_axis}]) <= {0.5 * muse_nfm.extent}"
) & pygad.ExprMask(f"abs(pos[:,2]) <= {0.5 * muse_nfm.extent}")
seeing = {"num": 25, "sigma": muse_nfm.pixel_width}

snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")

animation_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/misc/voronoi_sigma_only"

if False:
    for i, snapfile in enumerate(snapfiles):
        if i > 44:
            break
        print(f"Doing snapshot {i:03d}")
        fig, ax = plt.subplots(1, 2)
        fig.set_figwidth(1.5*fig.get_figwidth())

        snap = pygad.Snapshot(snapfile, physical=True)
        bgs.analysis.basic_snapshot_centring(snap)

        fig.suptitle(f"Time: {bgs.general.convert_gadget_time(snap):.2f} Gyr")

        # plot the density and save
        pygad.plotting.image(
            snap.stars[ifu_mask],
            qty="mass",
            xaxis=x_axis,
            yaxis=y_axis,
            scaleind="labels",
            ax=ax[0],
            vlim=(1e7, 5e9),
            Npx=muse_nfm.number_pixels,
            cbartitle=r"$\log(\Sigma/\mathrm{M}_\odot\mathrm{kpc}^{-2})$"
        )

        voronoi = bgs.analysis.VoronoiKinematics(
            x=snap.stars[ifu_mask]["pos"][:, x_axis],
            y=snap.stars[ifu_mask]["pos"][:, 2],
            V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
            m=snap.stars[ifu_mask]["mass"],
            Npx=muse_nfm.number_pixels,
            seeing=seeing
        )
        voronoi.make_grid(part_per_bin=5000 * seeing["num"])

        voronoi.binned_LOS_dispersion_only(ax=ax[1], clims=(170, 270), cbar="inset")
        ax[1].set_xlabel("x/kpc")
        ax[1].set_ylabel("z/kpc")

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