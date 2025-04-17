import argparse
import os.path
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

x_axis = 0
y_axis = 2
LOS_axis = 1

# set up the instruments
ifu_instr = bgs.analysis.MICADO_NFM()
ifu_instr.redshift = args.redshift
ifu_instr.max_extent = 1.5
SL.debug(f"Pixel width: {ifu_instr.pixel_width}")
seeing = {"num": 25, "sigma": ifu_instr.resolution_kpc}
SL.debug(f"IFU extent set to {ifu_instr.extent}")

# do this one snapshot
snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_009.hdf5"
#snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0720/output/snap_055.hdf5"
snap = pygad.Snapshot(snapfile, physical=True)
# centre on the BH
pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

ifu_mask = pygad.ExprMask(
    f"abs(pos[:,{x_axis}]) <= {0.5 * ifu_instr.extent}"
) & pygad.ExprMask(f"abs(pos[:,{y_axis}]) <= {0.5 * ifu_instr.extent}")

voronoi = bgs.analysis.VoronoiKinematics(
    x=snap.stars[ifu_mask]["pos"][:, x_axis],
    y=snap.stars[ifu_mask]["pos"][:, y_axis],
    V=snap.stars[ifu_mask]["vel"][:, LOS_axis],
    m=snap.stars[ifu_mask]["mass"],
    Npx=ifu_instr.number_pixels,
    seeing=seeing,
)
voronoi.make_grid(part_per_bin=int(25**2))
voronoi.binned_LOSV_statistics()
#bgs.utils.save_data(voronoi.dump_to_dict(), figure_config.data_path(f"micado_ifu_mock_{x_axis}{y_axis}.pickle"))

fig, ax = plt.subplots()
voronoi.plot_kinematic_maps(ax=ax, moments="2", cbar="inset")
ax.set_xticks([])
ax.set_yticks([])
bgs.plotting.draw_sizebar(ax=ax, length=0.5, units="kpc", size_vertical=0.02)

# add the BH location
#ax.scatter(snap.bh["pos"][:,x_axis], snap.bh["pos"][:,y_axis], marker="o", fc="k")
ax.annotate(
    r"$\mathrm{SMBH}$",
    (-0.02, 0),
    (-0.4, -0.4),
    color="k",
    arrowprops={"fc": "k", "ec": "k", "arrowstyle": "wedge"},
    ha="right",
    va="bottom"
)
'''
# add cluster radius
bound_ids = bgs.analysis.find_strongly_bound_particles(snap)
r = np.max(snap.stars[pygad.IDMask(bound_ids)]["r"])
cluster_rad = Circle(
    xy=(0, 0),
    radius=r,
    fc="none",
    ec="k",
    lw=1,
    ls=":",
)
ax.add_patch(cluster_rad)
theta_text = 135 * np.pi / 180
ax.annotate(
    r"$\mathrm{strongly\,bound\,stars}$",
    (r*np.cos(theta_text), r*np.sin(theta_text)),
    (-0.4, 0.4),
    color="k",
    arrowprops={"fc": "k", "ec": "k", "arrowstyle": "wedge"},
    ha="right",
    va="bottom"
)'''

bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/ifu_zoomed.png"), force_ext=True)
