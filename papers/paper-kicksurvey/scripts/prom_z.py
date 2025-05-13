import argparse
import os.path
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import baggins as bgs
from proj_density import ProjectedDensityObject
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot prominence as a function of redshift",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

if args.extract:
    # set a snapshot
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0420/output/snap_007.hdf5"
    # snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_009.hdf5"

    redshifts = np.linspace(0.1, 2, 50)
    proms = np.full_like(redshifts, np.nan)

    for i, z in enumerate(redshifts):
        SL.warning(f"Doing {z:.2f}")
        p = ProjectedDensityObject.load_single_snapshot(snapfile, redshift=z, logger=SL)
        save_figure = True if z > 1.9 or i == 0 else False
        p.run(save_figure=save_figure)
        proms[i] = p.cluster_prom
    data = {"z": redshifts, "proms": proms}
    bgs.utils.save_data(
        data,
        os.path.join(figure_config.reduced_data_dir, "prom_z.pickle"),
        exist_ok=True,
    )
else:
    data = bgs.utils.load_data(
        os.path.join(figure_config.reduced_data_dir, "prom_z.pickle")
    )
    redshifts = data["z"]
    proms = data["proms"]

fig, ax = plt.subplots()
ax.plot(redshifts, uniform_filter1d(proms, 4, mode="nearest"))
for pl in (2, 3, 4, 5):
    ax.axhline(pl, c="k", lw=1, zorder=0.3)
bgs.plotting.savefig(figure_config.fig_path("prom_z.pdf"), force_ext=True)
