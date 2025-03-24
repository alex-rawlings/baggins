import argparse
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Make IFU plot at different times",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-z", "--redshift", dest="redshift", type=float, help="redshift", default=0.6
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

ifu_snaps = [7, 14, 20, 45]
if args.extract:
    data = {"kv": args.kv}
    for s in ifu_snaps:
        data[f"snap_{s:03d}"] = {"t": None, "voronoi": None, "bh_pos": None}

snapfiles = bgs.utils.get_snapshots_in_dir(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output"
)
ketjufile = bgs.utils.get_ketjubhs_in_dir(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output"
)[0]

# set up interpolation arrays
t_interp = []
xcom_interp = []


muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = args.redshift
seeing = {"num": 25, "sigma": muse_nfm.pixel_width}
ifu_mask = pygad.ExprMask(f"abs(pos[:,0]) <= {0.5 * muse_nfm.extent}") & pygad.ExprMask(
    f"abs(pos[:,2]) <= {0.5 * muse_nfm.extent}"
)

for i, snapfile in enumerate(snapfiles[: max(ifu_snaps) + 1]):
    snap_num = int(bgs.general.get_snapshot_number(snapfile))
    if i % 3 == 0 or snap_num in ifu_snaps:
        SL.info(f"Reading snapshot {snap_num}")
        snap = pygad.Snapshot(snapfile, physical=True)
        # get the centre of mass
        xcom = pygad.analysis.shrinking_sphere(snap.stars, [0, 0, 0], 30)
        SL.debug(f"CoM is {xcom}")
        t_interp.append(bgs.general.convert_gadget_time(snap))
        xcom_interp.append(xcom)

        if args.extract and snap_num in ifu_snaps:
            data[f"snap_{snap_num:03d}"]["t"] = t_interp[-1]
            bgs.analysis.basic_snapshot_centring(snap)
            data[f"snap_{snap_num:03d}"]["bh_pos"] = [
                snap.bh["pos"][:, 0],
                snap.bh["pos"][:, 2],
            ]
            # create IFU plots
            voronoi = bgs.analysis.VoronoiKinematics(
                x=snap.stars[ifu_mask]["pos"][:, 0],
                y=snap.stars[ifu_mask]["pos"][:, 2],
                V=snap.stars[ifu_mask]["vel"][:, 1],
                m=snap.stars[ifu_mask]["mass"],
                Npx=muse_nfm.number_pixels,
                seeing=seeing,
            )
            voronoi.make_grid(part_per_bin=10000)
            voronoi.binned_LOSV_statistics()
            data[f"snap_{snap_num:03d}"]["voronoi"] = voronoi.dump_to_dict()

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
if args.extract:
    bgs.utils.save_data(data, figure_config.data_path("ifu_times.pickle"))
else:
    data = bgs.utils.load_data(figure_config.data_path("ifu_times.pickle"))
    assert args.kv == data["kv"]
ifu_ts = [v["t"] for k, v in data.items() if "snap" in k]

# initialise figure
fig, ax = plt.subplot_mosaic(
    """
    AAAAA
    BCDE.
    """,
    gridspec_kw={
        "width_ratios": [0.89, 0.89, 0.89, 1, 0.2],
        "wspace": 0.02,
        "height_ratios": [0.5, 1],
    },
)
fig.set_figwidth(2.2 * fig.get_figwidth())

# XXX top figure: BH position
# interpolate BH positions
SL.debug("Interpolating CoM positions...")
bh = bgs.analysis.get_bh_after_merger(ketjufile)
bh.t /= bgs.general.units.Gyr
bh.x /= bgs.general.units.kpc
xcom = np.full_like(bh.x, np.nan)
t_interp = np.asarray(t_interp)
xcom_interp = np.asarray(xcom_interp)
for i in range(3):
    xcom[:, i] = np.interp(bh.t, t_interp, xcom_interp[:, i])
r = bgs.mathematics.radial_separation(bh.x, xcom)
# sometimes we have a small bit at the start where the BH distance decreases as a result of interpolation, remove this
first_bit_bad = np.full_like(bh.t, True)
_counter = 0
while r[_counter] > r[_counter + 1]:
    first_bit_bad[_counter] = False
    _counter += 1
mask = np.logical_and(first_bit_bad, bh.t < 1.1 * max(ifu_ts))
p = ax["A"].plot(bh.t[mask], r[mask])
# add markers for IFU times
ifu_r = np.interp(ifu_ts, bh.t, r)
ax["A"].plot(ifu_ts, ifu_r, ls="", marker="o", c=p[-1].get_color(), mec="k", mew=0.5)
ax["A"].set_xlabel(r"$t/\mathrm{Gyr}$")
ax["A"].set_ylabel(r"$r/\mathrm{kpc}$")

clims = None
for k, v in data.items():
    if "snap" not in k:
        continue
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(v["voronoi"])
    if clims is None:
        clims = voronoi.get_colour_limits()
    else:
        _clims = voronoi.get_colour_limits()
        for k in _clims:
            if k == "sigma":
                clims[k][0] = np.min([clims[k][0], _clims[k][0]])
                clims[k][1] = np.max([clims[k][1], _clims[k][1]])
            else:
                clims[k] = max([clims[k], _clims[k]])

i = 0
for k, v in data.items():
    if "snap" not in k:
        continue
    axk = "BCDE"[i]
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(v["voronoi"])
    voronoi.plot_kinematic_maps(
        ax=ax[axk], moments="2", cbar=("adj" if i == 3 else ""), clims=clims
    )
    ax[axk].set_title(f"$t={v['t']:.2f}\,\mathrm{{Gyr}}$")
    ax[axk].set_xticks([])
    ax[axk].set_yticks([])
    bgs.plotting.draw_sizebar(ax[axk], 5, "kpc")
    ax[axk].scatter(
        v["bh_pos"][0],
        v["bh_pos"][1],
        lw=0.5,
        s=100,
        ec="k",
        fc="none",
    )
    i += 1

bgs.plotting.savefig(
    figure_config.fig_path(f"IFU_{args.kv:04d}_ortho.pdf"), force_ext=True
)
