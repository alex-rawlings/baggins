import argparse
import os.path
from datetime import datetime
import pygad
import cm_functions as cmf
import figure_config

parser = argparse.ArgumentParser(description="Plot core-kick relation given a Stan sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()


SL = cmf.setup_logger("script", args.verbosity)

snapfiles = cmf.utils.read_parameters(os.path.join(
                                            cmf.HOME,
                                            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml"
))

snapshots = dict()
for k, v in snapfiles["snap_nums"].items():
    if k == "v1020": break
    snapshots[k] = os.path.join(snapfiles["parent_dir"], f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5")


for k, v in snapshots.items():
    SL.info(f"Creating IFU maps for {k}")
    tstart = datetime.now()

    snap = pygad.Snapshot(v, physical=True)
    ball_mask = pygad.BallMask(
        R = pygad.analysis.half_mass_radius(snap.stars),
        center = pygad.analysis.shrinking_sphere(
            snap.stars,
            pygad.analysis.center_of_mass(snap.stars),
            30),
    )
    pygad.analysis.orientate_at(snap, "red I", total=True)
    voronoi_stats = cmf.analysis.voronoi_binned_los_V_statistics(
        snap.stars[ball_mask]["pos"][:,0],
        snap.stars[ball_mask]["pos"][:,1],
        snap.stars[ball_mask]["vel"][:,2],
        snap.stars[ball_mask]["mass"],
        Npx=50
        )

    SL.debug(f"Completed binning in {datetime.now()-tstart}")

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

    ax = cmf.plotting.voronoi_plot(voronoi_stats)
    fig = ax[0].get_figure()

    SL.info(f"Total time: {datetime.now() - tstart}")
    cmf.plotting.savefig(figure_config.fig_path(f"IFU/IFU_{k}.pdf"), force_ext=True)