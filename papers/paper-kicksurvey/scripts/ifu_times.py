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
ifu_ts = []
voronois = []
bh_pos = []

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

        if snap_num in ifu_snaps:
            ifu_ts.append(t_interp[-1])
            bgs.analysis.basic_snapshot_centring(snap)
            bh_pos.append([snap.bh["pos"][:, 0], snap.bh["pos"][:, 2]])
            # create IFU plots
            voronoi = bgs.analysis.VoronoiKinematics(
                x=snap.stars[ifu_mask]["pos"][:, 0],
                y=snap.stars[ifu_mask]["pos"][:, 2],
                V=snap.stars[ifu_mask]["vel"][:, 1],
                m=snap.stars[ifu_mask]["mass"],
                Npx=muse_nfm.number_pixels,
                seeing=seeing,
            )
            voronoi.make_grid(part_per_bin=5000 * seeing["num"])
            voronoi.binned_LOSV_statistics()
            voronois.append(voronoi)

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()

# initialise figure
fig, ax = plt.subplot_mosaic(
    """
    AAAAA
    BCDE.
    FGHI.
    """,
    gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.2], "wspace": 0.02},
)
fig.set_figwidth(2.2 * fig.get_figwidth())
fig.set_figheight(1.5 * fig.get_figheight())

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


# XXX IFU figures
def ax_generator(axm):
    for s in zip("BCDE", "FGHI"):
        yield np.array([axm[s[0]], axm[s[1]]])


ax_getter = ax_generator(ax)

clims = None
for voronoi in voronois:
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

for i, (voronoi, _bh, t) in enumerate(zip(voronois, bh_pos, ifu_ts)):
    this_ax = next(ax_getter)
    voronoi.plot_kinematic_maps(ax=this_ax, cbar=("adj" if i == 3 else ""), clims=clims)
    this_ax[0].set_title(f"$t={t:.2f}\,\mathrm{{Gyr}}$")
    for _ax in this_ax:
        _ax.set_xticks([])
        _ax.set_yticks([])
        bgs.plotting.draw_sizebar(_ax, 5, "kpc")
        _ax.scatter(
            _bh[0],
            _bh[1],
            lw=1,
            s=100,
            ec="k",
            fc="none",
        )

bgs.plotting.savefig(
    figure_config.fig_path(f"IFU_{args.kv:04d}_ortho.pdf"), force_ext=True
)
