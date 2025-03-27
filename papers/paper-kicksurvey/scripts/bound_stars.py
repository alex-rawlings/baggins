import argparse
import os.path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import baggins as bgs
import pygad
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot bound stellar mass",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-u", "--upper", help="upper velocity", type=float, dest="maxvel", default=1080
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

# XXX: set the data files and constant quantities we'll need
core_dispersion = 270  # km/s
core_radius = 0.58  # kpc
eff_radius = 5.65  # kpc
# apocentre data
apo_data_files = bgs.utils.get_files_in_dir(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data",
    ext=".txt",
)
# snapshot data
snapshot_dir = (
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
)
# main data file
main_data_dir = os.path.join(figure_config.reduced_data_dir, "bound_stars")

if args.extract:
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
        if np.any(np.diff(_r[apo_snap_num:]) > 0):
            # we have an instance where the distance of the BH to
            # centre is increasing
            peri_snap_num = np.nanargmin(_r[apo_snap_num:]) + apo_snap_num
        else:
            # we have run out of snapshots before pericentre
            SL.warning("No pericentre found, using first 10% of snapshots!")
            peri_snap_num = int(np.ceil(0.1 * len(_r)))
        if "0000" in file_name_only:
            # we have the special 0km/s case
            peri_snap_num = 10
            apo_snap_num = 5
        data = dict(
            t=np.full(peri_snap_num, np.nan),
            r=np.full(peri_snap_num, np.nan),
            Nbound=np.full(peri_snap_num, np.nan),
            rhalf=np.full(peri_snap_num, np.nan),
            other={},
        )

        start_kick_time = datetime.now()
        SL.info(f"Doing kick velocity {file_name_only.replace('kick-vel-','')}")
        for j, snapfile in tqdm(
            enumerate(
                bgs.utils.get_snapshots_in_dir(
                    os.path.join(snapshot_dir, file_name_only, "output")
                )[:peri_snap_num]
            ),
            desc="Analysing snapshots",
            total=peri_snap_num,
        ):
            snap = pygad.Snapshot(snapfile, physical=True)
            if j == 0:
                data["other"]["mstar"] = snap.stars["mass"][0]
                data["other"]["mbh"] = snap.bh["mass"][0]
                data["other"]["apo_snap_num"] = apo_snap_num
                data["other"]["vel"] = float(
                    os.path.splitext(file_name_only)[0].replace("kick-vel-", "")
                )
            if len(snap.bh) != 1:
                SL.warning("We require 1 BH! Skipping this snapshot")
                # clean memory
                snap.delete_blocks()
                del snap
                pygad.gc_full_collect()
                continue
            data["t"][j] = bgs.general.convert_gadget_time(snap, new_unit="Myr")
            bgs.analysis.basic_snapshot_centring(snap)
            data["r"][j] = snap.bh["r"].flatten()
            try:
                bound_ids = bgs.analysis.find_individual_bound_particles(snap)
                data["Nbound"][j] = len(bound_ids)
                bound_id_mask = pygad.IDMask(bound_ids)
                data["rhalf"][j] = float(
                    pygad.analysis.half_mass_radius(snap.stars[bound_id_mask])
                )
            except AssertionError as err:
                SL.exception(err)
                continue

            # clean memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()

        SL.warning(
            f"Completed extraction for {file_name_only} in {datetime.now()-start_kick_time}"
        )
        bgs.utils.save_data(
            data, os.path.join(main_data_dir, f"{file_name_only}-bound.pickle")
        )
        del data

# load the data files
data_files = bgs.utils.get_files_in_dir(main_data_dir, ".pickle")

# set up the figure
fig, ax = plt.subplots(1, 2)
fig.set_figwidth(2 * fig.get_figwidth())
vkcols = figure_config.VkickColourMap()
ax[0].set_xlabel(r"$t/t_{\mathrm{cross},\bullet}$")
ax[0].set_ylabel(r"$r/\mathrm{kpc}$")
ax[1].set_xlabel(r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$")
ax[1].set_ylabel(r"$M_\mathrm{bound}/\mathrm{M}_\odot$")
linthresh = 3
ax[0].set_yscale("symlog", linthresh=linthresh)

bp = None


def marker_size_scale(x):
    log_x = np.log10(x)
    x0 = (np.log10(5e5) + np.log10(5e9)) / 2  # centre in log10 space
    k = 2
    return 30 / (1 + np.exp(-k * (log_x - x0))) + 2


for i, df in enumerate(data_files):
    data = bgs.utils.load_data(df)
    # XXX temp fix
    vk = float(
        os.path.splitext(os.path.basename(df))[0]
        .replace("kick-vel-", "")
        .replace("-bound", "")
    )
    if i == 0:
        mstar = data["other"]["mstar"]
        m_bh = 2 * data["other"]["mbh"]  # BH yet to merge
    # need to find first pericentre
    try:
        apo_idx = np.nanargmax(data["r"])
        peri_idx = np.nanargmax(np.diff(data["r"][apo_idx:]) > 0) + apo_idx + 1
    except ValueError as err:
        SL.warning(f"Skipping: {vk}")
        SL.debug(err)
        continue

    # rescale time to 2 * crossing time: tc = R/v
    t = (data["t"][:peri_idx] - data["t"][~np.isnan(data["t"])][0]) / (
        data["t"][peri_idx] - data["t"][~np.isnan(data["t"])][0]
    )
    if vk <= args.maxvel:
        (lp,) = ax[0].plot(
            t, data["r"][:peri_idx], c=vkcols.get_colour(vk), ls="-", zorder=0.3, lw=1
        )
        # create marker sizes
        msizes = marker_size_scale(mstar * data["Nbound"][:peri_idx])
        ax[0].scatter(
            t,
            data["r"][:peri_idx],
            c=lp.get_color(),
            s=msizes,
            **figure_config.marker_kwargs,
        )
    if bp is None:
        (bp,) = ax[1].semilogy(
            vk, mstar * data["Nbound"][apo_idx], marker="o", ls="", mew="0.5", mec="k"
        )
    else:
        ax[1].semilogy(
            vk,
            mstar * data["Nbound"][apo_idx],
            marker="o",
            ls="",
            c=bp.get_color(),
            mew="0.5",
            mec="k",
        )
vkcols.make_cbar(ax[0])

# create the top legend for marker sizes
legend_masses = 10 ** np.arange(7, 9.1, 1)
# Create custom legend handles
legend_handles = [
    Line2D(
        [],
        [],
        marker="o",
        color="w",
        label=f"{np.log10(lm):.1f}",
        markerfacecolor="dimgray",
        markersize=np.sqrt(marker_size_scale(lm)),
    )
    for lm in legend_masses
]
# Add the legend above the plot
ax[0].legend(
    handles=legend_handles,
    title=r"$\log_{10}(M_\mathrm{bound}/\mathrm{M}_\odot)$",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=True,
)

# set regions to show core radius and effective radius
ylim = ax[0].get_ylim()
ax[0].axhspan(ylim[0], core_radius, fc="dimgray", alpha=0.6, zorder=0.1)
ax[0].text(0.3, 0.5 * core_radius, r"$r<r_{\mathrm{b},0}$")
ax[0].axhspan(ylim[0], eff_radius, fc="gray", alpha=0.4, zorder=0.1)
ax[0].text(0.3, 0.25 * eff_radius, r"$r<R_\mathrm{e}$")
ax[0].set_ylim(ylim)

# show core dispersion
xlim = ax[1].get_xlim()
ax[1].axvspan(xlim[0], core_dispersion, fc="dimgray", alpha=0.6, zorder=1)
ax[1].text(
    core_dispersion * 0.3,
    ax[1].get_ylim()[1] * 1e-3,
    r"$v_\mathrm{kick}< \sigma_{\star,0}$",
    rotation="vertical",
)
ax[1].set_xlim(xlim)

# set dual y axis on second plot
ax[1].tick_params(axis="y", which="both", right=False)
axr = ax[1].secondary_yaxis("right", functions=(lambda x: x / m_bh, lambda x: x * m_bh))
axr.set_ylabel(r"$M_\mathrm{bound}/M_\bullet$")


# Generate manual minor tick positions (log space, skipping linear region)
def log_minor_ticks(linthresh, max_exp=5):
    ticks = []
    # Positive side
    for exp in range(1, max_exp + 1):
        for sub in range(2, 10):
            val = sub * 10 ** (exp - 1)
            if val > linthresh:
                ticks.append(val)
    # Negative side (mirror)
    ticks += [-t for t in ticks]
    return sorted(ticks)


minor_ticks = log_minor_ticks(linthresh=np.floor(np.log10(linthresh)))

# Set manual minor ticks
ax[0].yaxis.set_minor_locator(FixedLocator(minor_ticks))

# Customize tick appearance
ax[0].tick_params(axis="y", which="minor", color="k")

bgs.plotting.savefig(figure_config.fig_path("bound.pdf"), force_ext=True)
