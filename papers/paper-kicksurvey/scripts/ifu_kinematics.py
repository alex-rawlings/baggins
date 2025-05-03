import argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import baggins as bgs
import pygad
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Make IFU plot at different times",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-z", "--redshift", dest="redshift", type=float, help="redshift", default=0.6
)
parser.add_argument(
    "--xy", dest="axes", help="position axes", type=int, nargs=2, default=[0, 2]
)
parser.add_argument(
    "-m", "--moment", type=str, help="moment to plot", dest="moment", default="2"
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

x_axis, y_axis = args.axes
LOS_axis = list(set({0, 1, 2}).difference({x_axis, y_axis}))[0]
muse_data_file = figure_config.data_path(f"ifu/muse_ifu_mock_{x_axis}{y_axis}.pickle")
harmoni_data_file = figure_config.data_path(
    f"ifu/harmoni_ifu_mock_{x_axis}{y_axis}.pickle"
)

# set up the instruments
# MUSE
muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = args.redshift
seeing = {"num": 25, "sigma": muse_nfm.resolution_kpc}
ifu_mask = pygad.ExprMask(
    f"abs(pos[:,{x_axis}]) <= {0.5 * muse_nfm.extent}"
) & pygad.ExprMask(f"abs(pos[:,{y_axis}]) <= {0.5 * muse_nfm.extent}")
# HARMONI
harmoni = bgs.analysis.HARMONI_SPATIAL()
harmoni.redshift = args.redshift
harmoni_extent = 1.5
harmoni.max_extent = harmoni_extent
seeing = {"num": 25, "sigma": harmoni.resolution_kpc}
harmoni_mask = pygad.ExprMask(
    f"abs(pos[:,{x_axis}]) <= {0.5 * harmoni.extent}"
) & pygad.ExprMask(f"abs(pos[:,{y_axis}]) <= {0.5 * harmoni.extent}")

SL.debug(muse_nfm)
SL.debug(harmoni)

# set the specific snaps to create IFU images for
ifu_snaps = {"0000": [2, 6, 40], "0540": [9, 16, 23], "0720": [22, 38, 55]}
harmoni_inset = [3, 8]

if args.extract:
    data_muse = dict.fromkeys(ifu_snaps, None)
    # have to this way so we don't have a self-referencing copy
    for k in data_muse.keys():
        data_muse[k] = {"t": [], "voronoi": [], "bhpos": [], "bhvel": []}
    data_harmoni = dict(voronoi=[], parts_per_bin=[60] * 9)

# fitted equation for theoretical detectability radius
core_sig = 270
core_rad = 0.58


# this is the threshold distance beyond which the cluster has a projected
# density above that of the background galaxy
def threshold_dist_theory(x):
    y = np.atleast_1d(1.18e-2 * x - 3.51)
    y[y < core_rad] = core_rad
    y[x < core_sig] = 1000
    return y


if args.extract:
    for i, (k, v) in enumerate(ifu_snaps.items()):
        SL.info(f"Doing kick velocity: {k}")
        snapfiles = bgs.utils.get_snapshots_in_dir(
            f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{int(k):04d}/output"
        )
        for j, snapnum in enumerate(v):
            SL.info(f"Doing snapshot {snapnum}")
            snap = pygad.Snapshot(snapfiles[snapnum], physical=True)
            bgs.analysis.basic_snapshot_centring(snap)
            data_muse[k]["t"].append(bgs.general.convert_gadget_time(snap))

            voronoi = bgs.analysis.VoronoiKinematics(
                x=snap.stars[ifu_mask]["pos"][:, x_axis],
                y=snap.stars[ifu_mask]["pos"][:, y_axis],
                V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
                m=snap.stars[ifu_mask]["mass"],
                Npx=muse_nfm.number_pixels,
                seeing=seeing,
            )
            voronoi.make_grid(part_per_bin=int(1000**2))
            voronoi.binned_LOSV_statistics()
            data_muse[k]["voronoi"].append(voronoi.dump_to_dict())
            data_muse[k]["bhpos"].append(
                [
                    deepcopy(snap.bh["pos"][:, x_axis].view(np.ndarray)),
                    deepcopy(snap.bh["pos"][:, y_axis].view(np.ndarray)),
                ]
            )
            data_muse[k]["bhvel"].append(
                [
                    deepcopy(snap.bh["vel"][:, x_axis].view(np.ndarray)),
                    deepcopy(snap.bh["vel"][:, y_axis].view(np.ndarray)),
                ]
            )

            if i * 3 + j in harmoni_inset:
                # do separate zoom for HARMONI
                # centre on the BH
                pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
                pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)
                assert np.allclose(
                    snap.bh["pos"].flatten(), np.zeros(3)
                ) and np.allclose(snap.bh["vel"].flatten(), np.zeros(3))
                voronoi = bgs.analysis.VoronoiKinematics(
                    x=snap.stars[harmoni_mask]["pos"][:, x_axis],
                    y=snap.stars[harmoni_mask]["pos"][:, y_axis],
                    V=snap.stars[harmoni_mask]["vel"][:, LOS_axis],
                    m=snap.stars[harmoni_mask]["mass"],
                    Npx=harmoni.number_pixels,
                    seeing=seeing,
                )
                voronoi.make_grid(
                    part_per_bin=data_harmoni["parts_per_bin"][i * 3 + j] ** 2
                )
                voronoi.binned_LOSV_statistics()
                data_harmoni["voronoi"].append(voronoi.dump_to_dict())
            else:
                data_harmoni["voronoi"].append([])

            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
    bgs.utils.save_data(data_muse, muse_data_file, exist_ok=True)
    bgs.utils.save_data(data_harmoni, harmoni_data_file, exist_ok=True)

data_muse = bgs.utils.load_data(muse_data_file)
data_harmoni = bgs.utils.load_data(harmoni_data_file)
if args.verbosity == "DEBUG":
    bgs.general.print_dict_summary(data_muse)

# set up the figure
fig, ax = plt.subplots(3, 3, sharex="all", sharey="all")
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(2 * fig.get_figheight())


# get the colour limits
def vor_generator(d):
    for v in d.values():
        for vv in v["voronoi"]:
            yield vv


vor_gen = vor_generator(data_muse)
clims = bgs.analysis.unify_IFU_colour_scheme(vor_gen)

# plot the data
dt = pygad.UnitScalar(2e-2, "Gyr")
visual_offset = pygad.UnitScalar(1e-3, "Gyr")
for i, (k, v) in enumerate(data_muse.items()):
    for j, (vor, bhx, bhv, t) in enumerate(
        zip(v["voronoi"], v["bhpos"], v["bhvel"], v["t"])
    ):
        SL.debug(f"Plotting {k} time {j}")
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xlim(-0.48 * muse_nfm.extent, 0.48 * muse_nfm.extent)
        ax[i, j].set_ylim(-0.48 * muse_nfm.extent, 0.48 * muse_nfm.extent)
        voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(vor)
        voronoi.plot_kinematic_maps(
            ax=ax[i, j], moments=args.moment, cbar="inset", clims=clims
        )
        ax[i, j].scatter(bhx[0], bhx[1], marker="o", fc="none", ec="k", lw=1)
        if float(k) > core_sig:
            ax[i, j].annotate(
                "",
                (bhx[0] + bhv[0] * dt, bhx[1] + bhv[1] * dt),
                (bhx[0] + bhv[0] * visual_offset, bhx[1] + bhv[1] * visual_offset),
                arrowprops=dict(arrowstyle="-|>", fc="k"),
            )
            # add theoretical detection radius
            detect_rad = Circle(
                xy=(0, 0),
                radius=threshold_dist_theory(float(k)),
                fc="none",
                ec="k",
                lw=1,
                ls=":",
            )
            ax[i, j].add_patch(detect_rad)
        bgs.plotting.draw_sizebar(
            ax=ax[i, j], length=10, units="kpc", size_vertical=0.5
        )
        ax[i, j].text(
            0.05, 0.9, f"${t:.3f}\,\mathrm{{Gyr}}$", transform=ax[i, j].transAxes
        )

for i, k in enumerate(data_muse.keys()):
    ax[i, 0].text(
        0.05,
        0.1,
        f"$v_\mathrm{{kick}}={float(k):.0f}\,\mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
        transform=ax[i, 0].transAxes,
    )

# add HARMONI inset panel
for hi, cbar_ticks in zip(harmoni_inset, ([200, 240], [200, 240])):
    SL.debug(f"Doing HARMONI inset {hi}")
    axins = ax.flatten()[hi].inset_axes(
        [0.53, 0.0, 0.45, 0.48],
        xticklabels=[],
        yticklabels=[],
    )
    axins.set_xticks([])
    axins.set_yticks([])
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(data_harmoni["voronoi"][hi])
    voronoi.plot_kinematic_maps(
        ax=axins,
        moments="2",
        cbar="inset",
        cbar_kwargs={"label": "", "aspect": 40, "ticks": cbar_ticks},
    )
    # XXX these need to be set by hand
    axins.set_xlim(-0.5, 0.5)
    axins.set_ylim(-0.3, 0.65)

    bhx, bhy = list(data_muse.values())[hi // 3]["bhpos"][hi % 3]

    ax.flatten()[hi].indicate_inset(
        bounds=[
            bhx[0] - harmoni_extent / 2,
            bhy[0] - harmoni_extent / 2,
            harmoni_extent,
            harmoni_extent,
        ],
        inset_ax=axins,
        alpha=1,
        edgecolor="k",
        lw=1,
    )
    axins.annotate(
        r"$\mathrm{BRC}$",
        (0, 0),
        (-0.1, 0.2),
        color="k",
        arrowprops={"fc": "k", "ec": "k", "arrowstyle": "wedge"},
        ha="right",
        va="bottom",
    )

bgs.plotting.savefig(figure_config.fig_path("IFU_mock.pdf"), force_ext=True)
