import argparse
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot bound stellar mass",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
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

# main data
data_file = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/kinematics_{args.kv:04d}.pickle"
data = bgs.utils.load_data(data_file)

# settling data
settle_data = np.loadtxt(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data/kick-vel-{args.kv:04d}.txt",
    skiprows=1,
)

# ketju_bhs data

# TODO use interpolation to get the time when v < sigma
# tsigma = np.interp(270, settle_data[:,2], settle_data[:,0])
tsigma = settle_data[bgs.general.get_idx_in_array(270, settle_data[:, 2][::-1]), 0]

# bound plot
star_mass = 5e4
bh_mass = 5.86e9
fig, ax = plt.subplots()
ax.tick_params(axis="y", which="both", right=False)
axr = ax.secondary_yaxis(
    "right", functions=(lambda x: x / bh_mass, lambda x: x * bh_mass)
)
ax.semilogy(
    data["time"],
    np.array(data["bound_stars_all"]) * star_mass,
    label=r"$M_\mathrm{bound}$",
)
ax.semilogy(
    data["time"],
    np.array(data["original_bound_stars"]) * star_mass,
    label=r"$M_\mathrm{bound,0}$",
)
ax.set_xlabel(r"$t/\mathrm{Gyr}$")
ax.set_ylabel(r"$M_\mathrm{bound}/M_\odot$")
axr.set_ylabel(r"$M_\mathrm{bound}/M_\bullet$")
ax.legend()
# ax.axvline(tsigma, ls=":", lw=1, c="k")
plt.subplots_adjust(left=0.2, top=0.95, bottom=0.15, right=0.8)
bgs.plotting.savefig(
    figure_config.fig_path(f"bound_{args.kv:04d}.pdf"), fig=fig, force_ext=True
)
