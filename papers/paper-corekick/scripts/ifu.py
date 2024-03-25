import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pygad
import baggins as bgs
import figure_config

parser = argparse.ArgumentParser(
    description="Plot IFU maps", formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

snapfiles = bgs.utils.read_parameters(
    os.path.join(
        bgs.HOME,
        "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
    )
)

snapshots = dict()
for k, v in snapfiles["snap_nums"].items():
    if k == "v1020":
        break
    snapshots[k] = os.path.join(
        snapfiles["parent_dir"], f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5"
    )

# dict to store radial h4 profiles
h4_vals = {}

# determine colour limits
Vlim = -np.inf
sigmalim = [np.inf, -np.inf]
h3lim = -np.inf
h4lim = -np.inf

for k, v in snapshots.items():
    if float(k[1:]) > 900:
        break
    SL.info(f"Creating IFU maps for {k}")
    tstart = datetime.now()

    snap = pygad.Snapshot(v, physical=True)
    pygad.analysis.orientate_at(snap, "red I", total=True)

    rhalf = pygad.analysis.half_mass_radius(snap.stars)

    ball_mask = pygad.BallMask(
        R=0.25 * rhalf,
        center=pygad.analysis.shrinking_sphere(
            snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
        ),
    )
    voronoi_stats = bgs.analysis.voronoi_binned_los_V_statistics(
        snap.stars[ball_mask]["pos"][:, 0],
        snap.stars[ball_mask]["pos"][:, 1],
        snap.stars[ball_mask]["vel"][:, 2],
        snap.stars[ball_mask]["mass"],
        part_per_bin=1000,
    )

    SL.debug(f"Completed binning in {datetime.now()-tstart}")

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

    # determine colour limits
    _Vlim = np.max(np.abs(voronoi_stats["img_V"]))
    Vlim = _Vlim if _Vlim > Vlim else Vlim

    _sigmalim = [np.min(voronoi_stats["img_sigma"]), np.max(voronoi_stats["img_sigma"])]
    sigmalim[0] = _sigmalim[0] if _sigmalim[0] < sigmalim[0] else sigmalim[0]
    sigmalim[1] = _sigmalim[1] if _sigmalim[1] > sigmalim[1] else sigmalim[1]

    _h3lim = np.max(np.abs(voronoi_stats["img_h3"]))
    h3lim = _h3lim if _h3lim > h3lim else h3lim

    _h4lim = np.max(np.abs(voronoi_stats["img_h4"]))
    h4lim = _h4lim if _h4lim > h4lim else h4lim

    # have to set colour limits by hand
    ax = bgs.plotting.voronoi_plot(
        voronoi_stats,
        clims={"V": [25], "sigma": [155, 351], "h3": [0.04], "h4": [0.05]},
    )
    fig = ax[0].get_figure()

    SL.info(f"Total time: {datetime.now() - tstart}")
    bgs.plotting.savefig(figure_config.fig_path(f"IFU/IFU_{k}.pdf"), force_ext=True)
    plt.close()

    h4_vals[k] = {}
    h4_vals[k]["R"], h4_vals[k]["h4"] = bgs.analysis.radial_profile_velocity_moment(
        voronoi_stats, "h4"
    )


# plot h4 radial profiles
fig, ax = plt.subplots(1, 1)
get_kick_val = lambda k: float(k.lstrip("v"))
kick_vels = [get_kick_val(k) for k in h4_vals.keys()]
cmapper, sm = bgs.plotting.create_normed_colours(
    vmin=min(kick_vels), vmax=max(kick_vels), cmap="custom_Blues"
)
for k, v in h4_vals.items():
    ax.plot(v["R"], v["h4"], c=cmapper(get_kick_val(k)), ls="-")
ax.set_xlabel(r"$R/\mathrm{kpc}$")
ax.set_ylabel(r"$\langle h_4 \rangle$")
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.set_ylabel(r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$")
bgs.plotting.savefig(figure_config.fig_path("h4.pdf"), force_ext=True)

print("-------------")
print("Colour limits")
print(f"V: {Vlim}")
print(f"sigma: {sigmalim}")
print(f"h3: {h3lim}")
print(f"h4: {h4lim}")
