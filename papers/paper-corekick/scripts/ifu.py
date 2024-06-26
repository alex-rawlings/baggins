import argparse
import os.path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from datetime import datetime
import pygad
import baggins as bgs
import figure_config

parser = argparse.ArgumentParser(
    description="Plot IFU maps", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
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

rhalf_factor = 0.25

h4_file = os.path.join(bgs.DATADIR, f"mergers/processed_data/core-paper-data/h4_{str(rhalf_factor).replace('.', '')}rhalf.pickle")

bgs.plotting.check_backend()

if args.extract:
    snapfiles = bgs.utils.read_parameters(
        os.path.join(
            bgs.HOME,
            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
        )
    )

    seeing = {"num": 25, "sigma": 0.3}

    snapshots = dict()
    for k, v in snapfiles["snap_nums"].items():
        if v is None:
            continue
        snapshots[k] = os.path.join(
            snapfiles["parent_dir"],
            f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5",
        )

    # dict to store radial h4 profiles
    h4_vals = {"para": {}, "ortho": {}, "rhalf":[]}

    # determine colour limits
    Vlim = -np.inf
    sigmalim = [np.inf, -np.inf]
    h3lim = -np.inf
    h4lim = -np.inf

    for k, v in snapshots.items():
        SL.info(f"Creating IFU maps for {k}")
        tstart = datetime.now()

        snap = pygad.Snapshot(v, physical=True)

        pre_ball_mask = pygad.BallMask(
            30,
            center=pygad.analysis.shrinking_sphere(
                snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
            ),
        )
        rhalf = pygad.analysis.half_mass_radius(snap.stars[pre_ball_mask])
        h4_vals["rhalf"].append(rhalf)
        extent = rhalf_factor * rhalf
        n_regular_bins = int(2 * extent / pygad.UnitScalar(0.04, "kpc"))

        '''ball_mask = pygad.BallMask(
            R=extent,
            center=pygad.analysis.shrinking_sphere(
                snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
            ),
        )'''
        box_mask = pygad.BoxMask(
            extent=2*extent,
            center=pygad.analysis.shrinking_sphere(
                snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
            )
        )

        SL.debug(f"IFU extent is {extent:.2f} kpc")
        SL.debug(f"Number of regular bins is {n_regular_bins}^2")

        # as the BH kick direction is always along the x-axis, use
        # this axis as the LOS parallel direction
        # try two orientations:
        # 1: LOS parallel with BH motion
        # 2: LOS perpendicular to BH motion
        for orientation, x_axis, LOS_axis in zip(("para", "ortho"), (1, 0), (0, 1)):
            SL.info(f"Doing {orientation} orientation...")
            voronoi_stats = bgs.analysis.voronoi_binned_los_V_statistics(
                x=snap.stars[box_mask]["pos"][:, x_axis],
                y=snap.stars[box_mask]["pos"][:, 2],
                V=snap.stars[box_mask]["vel"][:, LOS_axis],
                m=snap.stars[box_mask]["mass"],
                Npx=n_regular_bins,
                part_per_bin=2000 * seeing["num"],
                seeing=seeing,
            )

            SL.debug(f"Completed binning in {datetime.now()-tstart}")

            # determine colour limits
            _Vlim = np.max(np.abs(voronoi_stats["img_V"]))
            Vlim = _Vlim if _Vlim > Vlim else Vlim

            _sigmalim = [
                np.min(voronoi_stats["img_sigma"]),
                np.max(voronoi_stats["img_sigma"]),
            ]
            sigmalim[0] = _sigmalim[0] if _sigmalim[0] < sigmalim[0] else sigmalim[0]
            sigmalim[1] = _sigmalim[1] if _sigmalim[1] > sigmalim[1] else sigmalim[1]

            _h3lim = np.max(np.abs(voronoi_stats["img_h3"]))
            h3lim = _h3lim if _h3lim > h3lim else h3lim

            _h4lim = np.max(np.abs(voronoi_stats["img_h4"]))
            h4lim = _h4lim if _h4lim > h4lim else h4lim

            # have to set colour limits by hand
            ax = bgs.plotting.voronoi_plot(
                voronoi_stats,
                clims={"V": [39], "sigma": [160, 245], "h3": [0.041], "h4": [0.045]},
            )
            for axi in ax.flat:
                axi.set_xlim(-extent, extent)
                axi.set_ylim(-extent, extent)
            fig = ax[0].get_figure()

            SL.info(f"Total time: {datetime.now() - tstart}")
            bgs.plotting.savefig(
                figure_config.fig_path(f"IFU_{str(rhalf_factor).replace('.', '')}rhalf/IFU_{orientation}_{k}.pdf"), force_ext=True
            )
            plt.close()

            h4_vals[orientation][k] = {}
            (
                h4_vals[orientation][k]["R"],
                h4_vals[orientation][k]["h4"],
            ) = bgs.analysis.radial_profile_velocity_moment(voronoi_stats, "h4")

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()

    bgs.utils.save_data(h4_vals, h4_file)

    print("-------------")
    print("Colour limits")
    print(f"V: {Vlim}")
    print(f"sigma: {sigmalim}")
    print(f"h3: {h3lim}")
    print(f"h4: {h4lim}")
    print("-------------")
else:
    SL.warning(
        "Reading in saved dataset for h4 analysis, IFU maps will not be recreated!"
    )
    h4_vals = bgs.utils.load_data(h4_file)

# plot h4 radial profiles
fig, ax = plt.subplots(2, 1, sharex="all", sharey="all")
get_kick_val = lambda k: float(k.lstrip("v"))
vkcols = figure_config.VkickColourMap()

# helper function for plotting
def plot_helper(axi, k, v, rhalf):
    r = v["R"]
    h4 = v["h4"]
    idx_sorted = np.argsort(r)
    h4_filtered = gaussian_filter1d(h4[idx_sorted], 5, mode="nearest")
    axi.plot(
        r[idx_sorted],
        h4_filtered,
        c=vkcols.get_colour(get_kick_val(k)),
        ls="-"
    )

for rh, (kp, vp), (ko, vo) in zip(h4_vals["rhalf"], h4_vals["para"].items(), h4_vals["ortho"].items()):
    m = bgs.plotting.mplChars()[int(get_kick_val(kp)) // 600]
    plot_helper(ax[0], kp, vp, rh)
    plot_helper(ax[1], ko, vo, rh)
ax[-1].set_xlabel(r"$R/\mathrm{kpc}$")
ax[0].set_ylabel(r"$\langle h_4 \rangle\;\mathrm{(parallel)}$")
ax[1].set_ylabel(r"$\langle h_4 \rangle\;\mathrm{(orthogonal)}$")
vkcols.make_cbar(ax.flat)
bgs.plotting.savefig(figure_config.fig_path("h4.pdf"), force_ext=True)
