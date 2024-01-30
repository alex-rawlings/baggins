import argparse
import os.path
import matplotlib.pyplot as plt
from datetime import datetime
import pygad
import cm_functions as cmf
import figure_config

parser = argparse.ArgumentParser(description="Plot IFU maps", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

# dict to store radial h4 profiles
h4_vals = {}

for k, v in snapshots.items():
    if float(k[1:]) > 900: break
    SL.info(f"Creating IFU maps for {k}")
    tstart = datetime.now()

    snap = pygad.Snapshot(v, physical=True)
    pygad.analysis.orientate_at(snap, "red I", total=True)

    rhalf = pygad.analysis.half_mass_radius(snap.stars)

    ball_mask = pygad.BallMask(
        R = 0.25 * rhalf,
        center = pygad.analysis.shrinking_sphere(
            snap.stars,
            pygad.analysis.center_of_mass(snap.stars),
            30),
    )
    voronoi_stats = cmf.analysis.voronoi_binned_los_V_statistics(
        snap.stars[ball_mask]["pos"][:,0],
        snap.stars[ball_mask]["pos"][:,1],
        snap.stars[ball_mask]["vel"][:,2],
        snap.stars[ball_mask]["mass"],
        part_per_bin = 1000
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
    plt.close()

    h4_vals[k] = {}
    h4_vals[k]["R"], h4_vals[k]["h4"] = cmf.analysis.radial_profile_velocity_moment(voronoi_stats, "h4")
    h4_vals[k]["R"] /= rhalf


# plot h4 radial profiles
fig, ax = plt.subplots(1,1)
get_kick_val = lambda k: float(k.lstrip("v"))
kick_vels = [get_kick_val(k) for k in h4_vals.keys()]
cmapper, sm = cmf.plotting.create_normed_colours(vmin=min(kick_vels), vmax=max(kick_vels))
for k, v in h4_vals.items():
    ax.plot(v["R"], v["h4"], c=cmapper(get_kick_val(k)), ls="-")
ax.set_xlabel(r"$R/R_\mathrm{e}$")
ax.set_ylabel(r"$\langle h_4 \rangle$")
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.set_ylabel(r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$")
cmf.plotting.savefig(figure_config.fig_path("h4.pdf"), force_ext=True)

