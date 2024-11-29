import argparse
import os.path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import ketjugw
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

start_t = datetime.now()
# main data
data_file = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/kinematics_{args.kv:04d}.pickle"
data = bgs.utils.load_data(data_file)
plt.close()
# settling data
settle_data = np.loadtxt(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data/kick-vel-{args.kv:04d}.txt",
    skiprows=1,
)
# settled snapshots
bh_settled_idx = bgs.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml")["snap_nums"][f"v{args.kv:04d}"]

ketju_file = bgs.utils.get_ketjubhs_in_dir(f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output")[0]

SL.debug(f"Data loaded in {datetime.now() - start_t}")

# XXX Step 1: Determine when the BH is within the core
# find when the velocity falls below the velocity dispersion
tsigma = bgs.general.xval_of_quantity(270, settle_data[:,0], settle_data[:,2])
SL.debug(f"time when v < sigma permanently is {tsigma:.3e}")
t_sigma_idx = bgs.general.get_idx_in_array(tsigma, settle_data[:bh_settled_idx, 0])
SL.debug(f"Which has index {t_sigma_idx}")
# and then the BH stays within the core
tcore = bgs.general.xval_of_quantity(0.58, settle_data[t_sigma_idx:,0], settle_data[t_sigma_idx:,1])

# XXX Step 2: Identify pericentres
# only want those times before BH in core
bh = bgs.analysis.get_bh_after_merger(ketju_file)
tcore_idx = bgs.general.get_idx_in_array(tcore, bh.t/bgs.general.units.Gyr)
bh = bh[:tcore_idx]
# set t=0 to be the time of merger
bh.t = bh.t - bh.t[0]
ghost_particle = ketjugw.Particle(
    t=bh.t,
    m=np.zeros_like(bh.t),
    x=np.zeros((len(bh.t), 3)),
    v=np.zeros((len(bh.t), 3))
)

peri_times, peri_idxs, sep = bgs.analysis.find_pericentre_time(bh, ghost_particle, return_sep=True)
peri_times /= bgs.general.units.Gyr
SL.debug(f"Pericentre times are {peri_times}")
fig, ax = plt.subplots()
ax.set_xlabel(r"$t/\mathrm{Gyr}$")
ax.set_ylabel(r"$r/\mathrm{kpc}$")
ax.plot(bh.t / bgs.general.units.Gyr, sep / bgs.general.units.kpc, markevery=peri_idxs, marker="o")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"kicksurvey-study/sep-{args.kv}.png"), fig=fig)
plt.close()


# bound plot
star_mass = 5e4
bh_mass = 5.86e9
fig, ax = plt.subplots()
ax.tick_params(axis="y", which="both", right=False)
axr = ax.secondary_yaxis(
    "right", functions=(lambda x: x / bh_mass, lambda x: x * bh_mass)
)
t = np.array(data["time"]) - data["time"][0]
ax.semilogy(
    t,
    np.array(data["bound_stars_all"]) * star_mass,
    label=r"$M_\mathrm{bound}$",
)
ax.semilogy(
    t,
    np.array(data["original_bound_stars"]) * star_mass,
    label=r"$M_\mathrm{bound,0}$",
)
ax.set_xlabel(r"$t/\mathrm{Gyr}$")
ax.set_ylabel(r"$M_\mathrm{bound}/M_\odot$")
axr.set_ylabel(r"$M_\mathrm{bound}/M_\bullet$")
ax.legend()
for t in peri_times[:-1]:
    # the last pericentre calculation fails, so let's not plot it
    ax.axvline(t, ls=":", lw=1, c="k", zorder=0.1)
ax.text(tcore * 1.1, ax.get_ylim()[1]/300, r"$\mathrm{BH\;within\;core}$")
xlim = ax.get_xlim()
ax.axvspan(tcore, 1.5*xlim[1], fc="gray", alpha=0.4)
ax.set_xlim(xlim)
plt.subplots_adjust(left=0.2, top=0.95, bottom=0.15, right=0.8)
bgs.plotting.savefig(
    figure_config.fig_path(f"bound_{args.kv:04d}.pdf"), fig=fig, force_ext=True
)
bgs.plotting.savefig("bound_test.png")
