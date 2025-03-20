import argparse
import os.path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import ketjugw
import pygad
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

# XXX: set the data files we'll need
core_dispersion = 270  # km/s
# main data
data_file = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/kinematics_{args.kv:04d}_part_0.pickle"
data_specific = bgs.utils.load_data(data_file)
plt.close()
# settled snapshots
bh_settled_idx = bgs.utils.read_parameters(
    "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml"
)["snap_nums"][f"v{args.kv:04d}"]
# ketju file
ketju_file = bgs.utils.get_ketjubhs_in_dir(
    f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output"
)[0]
# apocentre data
apo_data_files = bgs.utils.get_files_in_dir(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data",
    ext=".txt",
)
# settling data
settle_data = np.loadtxt(
    os.path.join(apo_data_files, f"kick-vel-{args.kv:04d}.txt"),
    skiprows=1,
)
# snapshot data
snapshot_dir = (
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
)


SL.debug(f"Data loaded in {datetime.now() - start_t}")

# XXX Step 1: Determine when the BH is within the core
# find when the velocity falls below the velocity dispersion
tsigma = bgs.general.xval_of_quantity(
    core_dispersion, settle_data[:, 0], settle_data[:, 2]
)
SL.debug(f"time when v < sigma permanently is {tsigma:.3e}")
t_sigma_idx = bgs.general.get_idx_in_array(tsigma, settle_data[:bh_settled_idx, 0])
SL.debug(f"Which has index {t_sigma_idx}")
# and then the BH stays within the core
tcore = bgs.general.xval_of_quantity(
    0.58, settle_data[t_sigma_idx:, 0], settle_data[t_sigma_idx:, 1]
)

# XXX Step 2: Identify pericentres
# only want those times before BH in core
bh = bgs.analysis.get_bh_after_merger(ketju_file)
tcore_idx = bgs.general.get_idx_in_array(tcore, bh.t / bgs.general.units.Gyr)
bh = bh[:tcore_idx]
# set t=0 to be the time of merger
bh.t = bh.t - bh.t[0]
ghost_particle = ketjugw.Particle(
    t=bh.t,
    m=np.zeros_like(bh.t),
    x=np.zeros((len(bh.t), 3)),
    v=np.zeros((len(bh.t), 3)),
)

peri_times, peri_idxs, sep = bgs.analysis.find_pericentre_time(
    bh, ghost_particle, return_sep=True
)
peri_times /= bgs.general.units.Gyr
SL.debug(f"Pericentre times are {peri_times}")
fig, ax = plt.subplots()
ax.set_xlabel(r"$t/\mathrm{Gyr}$")
ax.set_ylabel(r"$r/\mathrm{kpc}$")
ax.plot(
    bh.t / bgs.general.units.Gyr,
    sep / bgs.general.units.kpc,
    markevery=peri_idxs,
    marker="o",
)
bgs.plotting.savefig(
    os.path.join(bgs.FIGDIR, f"kicksurvey-study/sep-{args.kv}.png"), fig=fig
)
plt.close()

# set up main plot
fig, ax = plt.subplots(1, 2)
fig.set_figwidth(2 * fig.get_figwidth())

# XXX Step 3: plot bound mass as a function of kick velocity
kick_velocities = np.full(len(apo_data_files), np.nan)
bound_mass = np.full_like(kick_velocities, np.nan)
snap_offset = 3
for i, apo_file in enumerate(apo_data_files):
    file_name_only = os.path.basename(apo_file).replace(".txt", "")
    _r = np.loadtxt(apo_file, skiprows=1)[snap_offset:, 1]
    if np.any(np.diff(_r) < 0):
        # we have an instance where the distance of the BH to
        # centre is decreasing
        apo_snap_num = np.nanargmax(_r) + snap_offset
    else:
        # no apocentre
        continue
    snap_file = bgs.utils.get_snapshots_in_dir(
        os.path.join(snapshot_dir, file_name_only, "output")
    )[apo_snap_num]
    SL.info(f"Reading snapshot {snap_file}")
    snap = pygad.Snapshot(snap_file, physical=True)
    if i == 0:
        star_mass = snap.stars["mass"][0]
        bh_mass = snap.bh["mass"][0]
    # get the bound mass within the influence radius of the BH
    bound_mass[i] = len(bgs.analysis.find_individual_bound_particles(snap)) * star_mass
    kick_velocities[i] = float(file_name_only.replace("kick-vel-", ""))
    # clean memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

for axi in ax:
    axi.tick_params(axis="y", which="both", right=False)
    axr = axi.secondary_yaxis(
        "right", functions=(lambda x: x / bh_mass, lambda x: x * bh_mass)
    )
    axr.set_ylabel(r"$M_\mathrm{bound}/M_\bullet$")

ax[0].set_yscale("log")
ax[0].scatter(kick_velocities, bound_mass, zorder=1.5, **figure_config.marker_kwargs)
ax[0].set_xlabel(r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$")
ax[0].set_ylabel(r"$M_\mathrm{bound}/\mathrm{M}_\odot$")
xlim = ax[0].get_xlim()
ax[0].axvspan(xlim[0], core_dispersion, fc="gray", alpha=0.4, zorder=1)
ax[0].text(
    core_dispersion * 0.3,
    ax[0].get_ylim()[1] * 1e-3,
    r"$v_\mathrm{kick}< \sigma_{\star,0}$",
    rotation="vertical",
)
ax[0].set_xlim(xlim)

# XXX Step 4: specific case kick velocity
t = np.array(data_specific["time"]) - data_specific["time"][0]
ax[1].semilogy(
    t,
    np.array(data_specific["bound_stars_all"]) * star_mass,
    label=r"$M_\mathrm{bound}$",
)
ax[1].semilogy(
    t,
    np.array(data_specific["original_bound_stars"]) * star_mass,
    label=r"$M_\mathrm{bound,0}$",
)
ax[1].set_xlabel(r"$t/\mathrm{Gyr}$")
ax[1].set_ylabel(r"$M_\mathrm{bound}/M_\odot$")

ax[1].legend()
for t in peri_times[:-1]:
    # the last pericentre calculation fails, so let's not plot it
    ax[1].axvline(t, ls=":", lw=1, c="k", zorder=0.1)
ax[1].text(tcore * 1.1, ax[1].get_ylim()[1] / 6, r"$\mathrm{BH\;within\;core}$")
xlim = ax[1].get_xlim()
ax[1].axvspan(tcore, 1.5 * xlim[1], fc="gray", alpha=0.4)
ax[1].set_xlim(xlim)
ax[1].set_ylim(1e8, ax[1].get_ylim()[1])
plt.subplots_adjust(left=0.2, top=0.95, bottom=0.15, right=0.8)
bgs.plotting.savefig(
    figure_config.fig_path(f"bound_{args.kv:04d}.pdf"), fig=fig, force_ext=True
)
