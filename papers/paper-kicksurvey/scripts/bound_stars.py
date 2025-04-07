import argparse
import os.path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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
fig, ax = plt.subplots(1, 1)
fig.set_figwidth(1.2 * fig.get_figwidth())
ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$")
ax.set_ylabel(r"$M_\mathrm{bound}/\mathrm{M}_\odot$")


def data_grabber():
    """
    Generator to get the data to plot

    Yields
    ------
    return_dict : dict
        plotting data
    """
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
        return_dict = {
            "vk": vk,
            "bound_a": np.nan,
            "bound_p": np.nan,
            "r_a": np.nan,
            "r_p": np.nan,
            "m_bh": m_bh,
        }
        try:
            apo_idx = np.nanargmax(data["r"])
            peri_idx = np.nanargmax(np.diff(data["r"][apo_idx:]) > 0) + apo_idx + 1
            return_dict["bound_a"] = mstar * data["Nbound"][apo_idx]
            return_dict["bound_p"] = mstar * data["Nbound"][peri_idx]
            return_dict["r_a"] = data["r"][apo_idx]
            return_dict["r_p"] = data["r"][peri_idx]
        except ValueError as err:
            SL.warning(f"Skipping: {vk}")
            SL.debug(err)
        yield return_dict


grab_data = data_grabber()
max_r = np.nanmax(list(d["r_a"] for d in grab_data))
SL.debug(f"Maximum radius is {max_r}")
r_col_mapper, sm = bgs.plotting.create_normed_colours(
    1e-3, max_r, cmap="rocket", norm="LogNorm"
)

grab_data = data_grabber()
reff_vel = None
for d in grab_data:
    if d["vk"] < 1 or d["vk"] > args.maxvel:
        m_bh = d["m_bh"]
        continue
    ax.semilogy([d["vk"]] * 2, [d["bound_a"], d["bound_p"]], lw=0.5, ls="-", c="k")
    ax.semilogy(
        d["vk"],
        d["bound_p"],
        marker="s",
        ls="",
        mew="0.5",
        mec="k",
        c=r_col_mapper(d["r_p"]),
    )
    ax.semilogy(
        d["vk"],
        d["bound_a"],
        marker="o",
        ls="",
        mew="0.5",
        mec="k",
        c=r_col_mapper(d["r_a"]),
    )
    if d["r_a"] > eff_radius and reff_vel is None:
        reff_vel = d["vk"]
plt.colorbar(sm, ax=ax, label=r"$r/\mathrm{kpc}$", location="top")

ax.scatter(
    [], [], marker="o", c="gray", lw="0.5", ec="k", label=r"$\mathrm{apocentre}$"
)
ax.scatter(
    [], [], marker="s", c="gray", lw="0.5", ec="k", label=r"$\mathrm{pericentre}$"
)
ax.legend()

# set dual y axis on second plot
ax.tick_params(axis="y", which="both", right=False)
axr = ax.secondary_yaxis("right", functions=(lambda x: x / m_bh, lambda x: x * m_bh))
axr.set_ylabel(r"$M_\mathrm{bound}/M_\bullet$")

# show core dispersion
xlim = ax.get_xlim()
ax.axvspan(
    xlim[0], core_dispersion, zorder=1, hatch="//", fc="none", ec="dimgray", lw=1
)
ax.text(
    0.1,
    0.5,
    r"$v_\mathrm{kick}< \sigma_{\star,0}$",
    rotation="vertical",
    transform=ax.transAxes,
    va="center",
    bbox={"fc": "w", "ec": "none"},
)

# show were r_apo > Re
ax.axvspan(reff_vel, xlim[1], alpha=0.6, zorder=1, fc="gray")
ax.text(
    1.1 * reff_vel,
    1e7,
    r"$r_\mathrm{apo} > R_\mathrm{e}$",
)
ax.set_xlim(xlim)

bgs.plotting.savefig(figure_config.fig_path("bound.pdf"), force_ext=True)
