import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import baggins as bgs
import pygad
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Extract kinematic quantities",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(dest="orientation", choices=["para", "ortho"], help="orientation")
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
)
parser.add_argument(
    "-z", "--redshift", dest="redshift", type=float, help="redshift", default=0.3
)
parser.add_argument(
    "-a", "--animate", action="store_true", dest="animate", help="make animation"
)
parser.add_argument(
    "-f",
    "--final",
    dest="final",
    help="final snapshot number to make",
    default=None,
    type=int,
)
parser.add_argument(
    "--stride", type=int, help="plot every nth snapshot", dest="stride", default=None
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)


def make_data_file(n):
    return f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/kinematics_{args.kv:04d}_part_{n}.pickle"


snapdir = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output"
animation_path = (
    "/scratch/pjohanss/arawling/collisionless_merger/visualisations/recoil-explore"
)
final_snap = bgs.utils.read_parameters(
    "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml"
)["snap_nums"][f"v{args.kv:04d}"]
if args.final is not None:
    final_snap = args.final
save_data_interval = min([80, final_snap])
SL.warning(f"Data will be saved every {save_data_interval} snapshots...")


def make_data_dict():
    return dict(
        redshift=args.redshift,
        time=[],
        rhalf=[],
        bound_stars_all=[],
        original_bound_stars=[],
        para=dict(ifu=[], bh_pos_x=[], bh_pos_y=[], dens=[]),
        ortho=dict(ifu=[], bh_pos_x=[], bh_pos_y=[], dens=[]),
    )


# set the IFU instrument properties
muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = args.redshift

if args.extract:
    seeing = {"num": 25, "sigma": muse_nfm.pixel_width}
    data = make_data_dict()

    snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
    if args.stride is not None:
        assert args.stride > 1
        snapfiles = snapfiles[:: args.stride]
    first_snap_done = False
    for i, snapfile in enumerate(snapfiles):
        snap = pygad.Snapshot(snapfile, physical=True)
        snapnum = int(
            os.path.splitext(os.path.basename(snapfile).replace("snap_", ""))[0]
        )
        if len(snap.bh) > 1:
            SL.warning(f"Two BHs present in snapshot {i} -> skipping")
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            continue
        if snapnum > final_snap:
            SL.warning("Final snapshot: breaking")
            break
        SL.info(f"Doing snapshot {snapnum:03d}")
        snap_time = bgs.general.convert_gadget_time(snap)
        # we don't need all the snapshots as the BH is thermalising
        if snap_time > 1 and i % 4 != 0:
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            continue
        data["time"].append(snap_time)

        # move to CoM frame
        SL.debug(f"BH is at {snap.bh['pos']}")
        bgs.analysis.basic_snapshot_centring(snap)
        SL.debug(f"BH is now position: {snap.bh['pos']}")

        galaxy_mask = pygad.BallMask(30)
        rhalf = pygad.analysis.half_mass_radius(snap.stars[galaxy_mask])
        data["rhalf"].append(rhalf)
        if not first_snap_done:
            bound_stars_0 = bgs.analysis.find_individual_bound_particles(snap)
            data["bound_stars_all"].append(len(bound_stars_0))
            data["original_bound_stars"].append(len(bound_stars_0))
            first_snap_done = True
        else:
            try:
                bound_stars = bgs.analysis.find_individual_bound_particles(snap)
                data["bound_stars_all"].append(len(bound_stars))
                data["original_bound_stars"].append(
                    len(set(bound_stars_0).intersection(set(bound_stars)))
                )
            except AssertionError:
                data["bound_stars_all"].append(np.nan)
                data["original_bound_stars"].append(np.nan)
        for orientation, x_axis, LOS_axis in zip(("para", "ortho"), (1, 0), (0, 1)):
            ifu_mask = pygad.ExprMask(
                f"abs(pos[:,{x_axis}]) <= {0.5 * muse_nfm.extent}"
            ) & pygad.ExprMask(f"abs(pos[:,2]) <= {0.5 * muse_nfm.extent}")

            data[orientation]["bh_pos_x"].append(snap.bh["pos"][:, x_axis])
            data[orientation]["bh_pos_y"].append(snap.bh["pos"][:, 2])

            voronoi = bgs.analysis.VoronoiKinematics(
                x=snap.stars[ifu_mask]["pos"][:, x_axis],
                y=snap.stars[ifu_mask]["pos"][:, 2],
                V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
                m=snap.stars[ifu_mask]["mass"],
                Npx=muse_nfm.number_pixels,
                seeing=seeing,
            )
            voronoi.make_grid(part_per_bin=5000 * seeing["num"])
            voronoi.binned_LOSV_statistics()
            data[orientation]["ifu"].append(voronoi.dump_to_dict())

            # plot the density and save
            _fig, _ax, _aximage, _cbar = pygad.plotting.image(
                snap.stars[ifu_mask],
                qty="mass",
                xaxis=x_axis,
                yaxis=2,
                scaleind="labels",
            )
            data[orientation]["dens"].append(_aximage)

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
        del voronoi

        if i % save_data_interval == 0:
            SL.warning("Dumping data")
            bgs.utils.save_data(
                data, make_data_file(i // save_data_interval - 1), exist_ok=True
            )
            data = make_data_dict()
else:
    data_file = make_data_file(0)
    SL.warning(f"Reading from {data_file}")
    data = bgs.utils.load_data(data_file)

N_frames = len(data["time"])


# set axis and colour limits
def voronoi_colour_limit_maker(cvals):
    return dict(
        max_V=max([np.max(np.abs(v["img_V"])) for v in cvals]),
        max_s=max([np.max(v["img_sigma"]) for v in cvals]),
        min_s=min([np.min(v["img_sigma"]) for v in cvals]),
        max_h3=max([np.max(np.abs(v["img_h3"])) for v in cvals]),
        max_h4=max([np.max(np.abs(v["img_h4"])) for v in cvals]),
    )


def make_plot_and_save_for_gif(i, max_dens, min_dens):
    fig, ax = plt.subplot_mosaic(
        """
    ABE
    CDF
    """,
        figsize=(8, 4),
    )

    fig.suptitle(f"{data['time'][i]-data['time'][0]:.3f} Gyr")

    # get the colour limits
    clims = voronoi_colour_limit_maker(data[args.orientation]["ifu"])

    # voronoi plots
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(
        data[args.orientation]["ifu"][i]
    )
    voronoi.plot_kinematic_maps(
        ax=np.array([ax["A"], ax["B"], ax["C"], ax["D"]]),
        clims={
            "V": [clims["max_V"]],
            "sigma": [clims["min_s"], clims["max_s"]],
            "h3": [clims["max_h3"]],
            "h4": [clims["max_h4"]],
        },
    )
    for k in "ABCD":
        ax[k].scatter(
            data[args.orientation]["bh_pos_x"][i],
            data[args.orientation]["bh_pos_y"][i],
            lw=2,
            s=100,
            ec="green",
            fc="none",
        )
        ax[k].set_xlim(-0.5 * muse_nfm.extent, 0.5 * muse_nfm.extent)
        ax[k].set_ylim(-0.5 * muse_nfm.extent, 0.5 * muse_nfm.extent)
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
        ax[k].set_ylabel("z/kpc")

    # density plots
    dens = data[args.orientation]["dens"][i]
    for k in "EF":
        ax[k].imshow(
            dens.get_array(),
            extent=dens.get_extent(),
            cmap=dens.get_cmap(),
            vmax=max_dens,
            vmin=min_dens,
            origin="lower",
        )
        # make an "aperture" rectangle to show IFU footprint
        ifu_rect = Rectangle(
            (-0.5 * muse_nfm.extent, -0.5 * muse_nfm.extent),
            muse_nfm.extent,
            muse_nfm.extent,
            fc="none",
            ec="k",
            fill=False,
        )
        ax[k].add_artist(ifu_rect)
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")
        ax[k].set_ylabel("z/kpc")
    ax["E"].scatter(
        data[args.orientation]["bh_pos_x"][i],
        data[args.orientation]["bh_pos_y"][i],
        lw=2,
        s=100,
        ec="red",
        fc="none",
    )
    # TODO add contours

    # fig.tight_layout()
    os.makedirs(os.path.join(animation_path, args.orientation), exist_ok=True)
    bgs.plotting.savefig(
        os.path.join(animation_path, args.orientation, f"frame_ifu_{i:02d}.png"),
        fig=fig,
    )


if args.animate:
    max_dens = max([np.nanmax(im.get_array()) for im in data[args.orientation]["dens"]])
    min_dens = min([np.nanmin(im.get_array()) for im in data[args.orientation]["dens"]])
    # make the gif
    for i in range(N_frames):
        make_plot_and_save_for_gif(i, max_dens=max_dens, min_dens=min_dens)
    frames = bgs.utils.get_files_in_dir(
        os.path.join(animation_path, args.orientation), ".png"
    )

    frames = [Image.open(img) for img in frames if "frame_ifu_" in img]  # Load images
    frames.extend([frames[-1]] * 10)

    # Save as an animated GIF
    frames[0].save(
        os.path.join(
            animation_path,
            args.orientation,
            f"animation_{args.orientation}_{args.kv:04d}.gif",
        ),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0,
    )

    plt.close()
else:
    # make still panel of select times
    fig, ax = plt.subplot_mosaic(
        """
    AAAAA
    BCDE.
    FGHI.
    """,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.2], "wspace": 0.02},
    )
    fig.set_figwidth(2.5 * fig.get_figwidth())
    fig.set_figheight(1.5 * fig.get_figheight())
    for k in "CDEFGHI":
        ax[k].sharex(ax["B"])
        ax[k].sharey(ax["B"])
    for k in "BF":
        ax[k].set_ylabel("z/kpc")
    for k in "FGHI":
        ax[k].set_xlabel(f"{'y' if args.orientation=='para' else 'x'}/kpc")

    # set the specific snapshots to plot
    specific_snaps = dict(ortho=[7, 14, 20, 45], para=[0, 2, 18, 35])
    data_subset = [
        v
        for i, v in enumerate(data[args.orientation]["ifu"])
        if i in specific_snaps[args.orientation]
    ]
    clims = voronoi_colour_limit_maker(data_subset)

    # BH position plot
    t = np.array(data["time"]) - data["time"][0]
    r = np.sqrt(
        np.array(data[args.orientation]["bh_pos_x"]) ** 2
        + np.array(data[args.orientation]["bh_pos_y"]) ** 2
    )
    ax["A"].plot(t, r, marker="o", markevery=specific_snaps[args.orientation])
    ax["A"].set_xlabel("t/Gyr")
    ax["A"].set_ylabel("r/kpc")

    # voronoi plots
    for i, vax, sax, tt in zip(
        specific_snaps[args.orientation],
        "BCDE",
        "FGHI",
        t[specific_snaps[args.orientation]],
    ):
        bgs.plotting.voronoi_plot(
            data[args.orientation]["ifu"][i],
            clims={
                "V": [clims["max_V"]],
                "sigma": [clims["min_s"], clims["max_s"]],
                "h3": [clims["max_h3"]],
                "h4": [clims["max_h4"]],
            },
            ax=np.array([ax[vax], ax[sax]]),
            cbar="adj" if vax == "E" else "",
        )
        ax[vax].set_title(f"$t={tt:.2f}\,\mathrm{{Gyr}}$")
        for k in (vax, sax):
            ax[k].scatter(
                data[args.orientation]["bh_pos_x"][i],
                data[args.orientation]["bh_pos_y"][i],
                lw=1.5,
                s=100,
                ec="green",
                fc="none",
            )
            ax[k].set_xticks([])
            ax[k].set_yticks([])
            bgs.plotting.draw_sizebar(ax[k], 5, "kpc")
    bgs.plotting.savefig(
        figure_config.fig_path(f"IFU_panel_{args.kv:04d}_{args.orientation}.pdf"),
        fig=fig,
        force_ext=True,
    )
