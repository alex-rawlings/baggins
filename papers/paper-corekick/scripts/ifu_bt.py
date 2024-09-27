import argparse
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use
    use("Agg")
    import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import baggins as bgs
import pygad
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot IFU maps", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    dest="vel",
    help="Velocity to plot"
)
parser.add_argument("-I",
                    "--Inertia",
                    action="store_true",
                    help="align with inertia",
                    dest="inertia")
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

# XXX set the fraction of rhalf within which IFU maps are created for
rhalf_factor = 0.25
seeing = {"num": 25, "sigma": 0.3}

# get the snapshot, all this is taken from IFU.py
snapfiles = bgs.utils.read_parameters(
    os.path.join(
        bgs.HOME,
        "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml"
    )
)
snapshots = dict()
for k, v in snapfiles["snap_nums"].items():
    if v is None:
        continue
    snapshots[k] = os.path.join(
        snapfiles["parent_dir"],
        f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5",
    )
snap = pygad.Snapshot(snapshots[f"v{args.vel}"], physical=True)
centre = pygad.analysis.shrinking_sphere(
        snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
    )
pre_ball_mask = pygad.BallMask(
    30,
    center=centre,
)
rhalf = pygad.analysis.half_mass_radius(snap.stars[pre_ball_mask])
extent = rhalf_factor * rhalf
n_regular_bins = int(2 * extent / pygad.UnitScalar(0.04, "kpc"))

if args.inertia:
    pygad.Translation(-centre).apply(snap, total=True)
    pygad.analysis.orientate_at(snap.stars[pre_ball_mask], "red I", total=True)

box_mask = pygad.BoxMask(
    extent=2 * extent,
    center=pygad.analysis.shrinking_sphere(
        snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
    ),
)

# determine the correct orbit file
orbitfilebase = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name and args.vel in d.name
][0]
SL.info(f"Reading: {orbitfilebase}")
orbitcl = bgs.utils.get_files_in_dir(orbitfilebase, ext=".cl", recursive=True)[0]

# get the core radius
core_radius = np.nanmedian(
    bgs.utils.load_data("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle")["rb"][args.vel].flatten()
)
SL.debug(f"Using a core radius of {core_radius:.2e}")

# read in orbit classification data
mergemask = bgs.analysis.MergeMask()
mergemask.add_family("box", [1,5,9,13,17,21,24,25], r"$\mathrm{box}$")
mergemask.add_family("tube", [4,8,12,16,20,3,7,11,15,19,2,6,10,14,18,26], r"$\mathrm{tube}$")
classifier = bgs.analysis.OrbitClassifier(orbitcl, mergemask=mergemask)

# now we will create three sets of IFU maps: one for all, one for tubes, one for boxes
fig, ax = plt.subplots(3, 4, sharex="all", sharey="all")
fig.set_figwidth(3 * fig.get_figwidth())
fig.set_figheight(1.5 * fig.get_figheight())
ax[0, 0].text(0.1, 0.9, "all", ha="left", va="center", transform=ax[0,0].transAxes)
ax[1,0].text(0.1, 0.9, "box", ha="left", va="center", transform=ax[1,0].transAxes)
ax[2,0].text(0.1, 0.9, "tube", ha="left", va="center", transform=ax[2,0].transAxes)
for axi in ax[-1,:]:
    axi.set_xlabel(r"$y/\mathrm{kpc}$")
for axi in ax[:,0]:
    axi.set_ylabel(r"$z/\mathrm{kpc}$")
ax[0,0].set_xlim(-2.6, 2.6)
ax[0,0].set_ylim(-2.6, 2.6)
# XXX set colour limits manually
if args.inertia:
    clims = {"V":[15], "sigma":[210, 320], "h3":[0.02], "h4":[0.065]}
else:
    clims = {"V":[16.1], "sigma":[225, 275], "h3":[0.028], "h4":[0.028]}


# all orbits
voronoi_stats_all = bgs.analysis.voronoi_binned_los_V_statistics(
    x=snap.stars[box_mask]["pos"][:, 1],
    y=snap.stars[box_mask]["pos"][:, 2],
    V=snap.stars[box_mask]["vel"][:, 0],
    m=snap.stars[box_mask]["mass"],
    Npx=n_regular_bins,
    part_per_bin=2000 * seeing["num"],
    seeing=seeing,
)
bgs.plotting.voronoi_plot(voronoi_stats_all, ax=ax[0,:], clims=clims)

# box orbits
mask = pygad.IDMask(classifier.get_particle_ids_for_family("box")) & box_mask
voronoi_stats_box = bgs.analysis.voronoi_binned_los_V_statistics(
    x=snap.stars[mask]["pos"][:, 1],
    y=snap.stars[mask]["pos"][:, 2],
    V=snap.stars[mask]["vel"][:, 0],
    m=snap.stars[mask]["mass"],
    Npx=n_regular_bins,
    part_per_bin=2000 * seeing["num"],
    seeing=seeing,
)
bgs.plotting.voronoi_plot(voronoi_stats_box, ax=ax[1,:], clims=clims)

# tube orbits
mask = pygad.IDMask(classifier.get_particle_ids_for_family("tube")) & box_mask
voronoi_stats_tube = bgs.analysis.voronoi_binned_los_V_statistics(
    x=snap.stars[mask]["pos"][:, 1],
    y=snap.stars[mask]["pos"][:, 2],
    V=snap.stars[mask]["vel"][:, 0],
    m=snap.stars[mask]["mass"],
    Npx=n_regular_bins,
    part_per_bin=2000 * seeing["num"],
    seeing=seeing,
)
bgs.plotting.voronoi_plot(voronoi_stats_tube, ax=ax[2,:], clims=clims)

# add the core radius to all plots
for i, axi in enumerate(ax.flat):
    core_circle = Circle((0, 0), core_radius, fill=False, ec=("w" if i%4==1 else "k"), ls="--")
    axi.add_artist(core_circle)

plt.subplots_adjust(left=0.03, right=0.95, top=0.97)
suffix = "_I" if args.inertia else ""
bgs.plotting.savefig(figure_config.fig_path(f"IFU_bt_{args.vel}{suffix}.pdf"), force_ext=True)

# print colour information
SL.info(f"Max V is {np.max([np.max(np.abs(v['img_V'])) for v in [voronoi_stats_all, voronoi_stats_box, voronoi_stats_tube]]):.2e}")
SL.info(f"Min sigma is {np.min([np.max(v['img_sigma']) for v in [voronoi_stats_all, voronoi_stats_box, voronoi_stats_tube]]):.2e}")
SL.info(f"Max sigma is {np.max([np.max(v['img_sigma']) for v in [voronoi_stats_all, voronoi_stats_box, voronoi_stats_tube]]):.2e}")
SL.info(f"Max h3 is {np.max([np.max(np.abs(v['img_h3'])) for v in [voronoi_stats_all, voronoi_stats_box, voronoi_stats_tube]]):.2e}")
SL.info(f"Max h4 is {np.max([np.max(np.abs(v['img_h4'])) for v in [voronoi_stats_all, voronoi_stats_box, voronoi_stats_tube]]):.2e}")
