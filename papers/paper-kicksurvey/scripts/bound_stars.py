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
            t=np.full(2, np.nan),
            r=np.full(2, np.nan),
            bound_mass=np.full(2, np.nan),
            ids=[[], []],
            rhalf=np.full(2, np.nan),
            other={},
            ambient_sigma=np.full(2, np.nan),
        )

        start_kick_time = datetime.now()
        SL.info(f"Doing kick velocity {file_name_only.replace('kick-vel-','')}")
        snapfiles = (
            bgs.utils.get_snapshots_in_dir(
                os.path.join(snapshot_dir, file_name_only, "output")
            )[sn]
            for sn in [apo_snap_num, peri_snap_num]
        )
        for j, snapfile in tqdm(
            enumerate(snapfiles),
            desc="Analysing snapshots",
            total=2,
        ):
            snap = pygad.Snapshot(snapfile, physical=True)
            if len(snap.bh) != 1:
                raise RuntimeError("We require 1 BH! Skipping this snapshot")
            if j == 0:
                data["other"]["mstar"] = snap.stars["mass"][0]
                data["other"]["mbh"] = snap.bh["mass"][0]
                data["other"]["vel"] = float(
                    os.path.splitext(file_name_only)[0].replace("kick-vel-", "")
                )
            data["t"][j] = bgs.general.convert_gadget_time(snap, new_unit="Myr")
            bgs.analysis.basic_snapshot_centring(snap)
            data["r"][j] = snap.bh["r"].flatten()
            try:
                strong_bound_ids, energy, amb_sigma = (
                    bgs.analysis.find_strongly_bound_particles(snap, return_extra=True)
                )
                data["ambient_sigma"][j] = amb_sigma
                bound_id_mask = pygad.IDMask(strong_bound_ids)
                data["bound_mass"][j] = np.sum(snap.stars[bound_id_mask]["mass"])
                data["ids"][j] = strong_bound_ids
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
ax.set_ylabel(r"$M/\mathrm{M}_\odot$")


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
            m_bh = data["other"]["mbh"]
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
            return_dict["bound_a"] = data["bound_mass"][0]
            return_dict["bound_p"] = data["bound_mass"][1]
            return_dict["r_a"] = data["r"][0]
            return_dict["r_p"] = data["r"][1]
        except ValueError as err:
            SL.warning(f"Skipping: {vk}")
            SL.debug(err)
        diff_ids = list(set(data["ids"][0]).difference(set(data["ids"][1])))
        SL.debug(
            f"{len(diff_ids)/len(data['ids'][0]):.3f} of particles are different between apo and peri centres"
        )
        yield return_dict


def load_obs_cluster_data():
    """
    Load the mass data calculated from perfect_observability.py

    Yields
    ------
    : tuple
        kick velocity and apocentre mass
    """
    dat_files = bgs.utils.get_files_in_dir(
        os.path.join(figure_config.reduced_data_dir, "perfect_obs"),
        ".pickle",
    )
    for f in dat_files:
        vk = float(os.path.splitext(os.path.basename(f))[0].replace("perf_obs_", ""))
        if vk > args.maxvel:
            continue
        cluster = bgs.utils.load_data(f)["cluster_props"]
        m = cluster[-1]["cluster_mass"]
        apo = cluster[-1]["r_centres_cluster"][0]
        if m is None:
            m = np.nan
        yield vk, m, apo


grab_data = data_grabber()
max_r = np.nanmax(list(d["r_a"] for d in grab_data))
SL.debug(f"Maximum radius is {max_r}")
r_col_mapper, sm = bgs.plotting.create_normed_colours(
    1e-3, max_r, cmap="rocket", norm="LogNorm"
)

# XXX plot the strongly bound mass
grab_data = data_grabber()
reff_vel = None
for d in grab_data:
    if d["vk"] < 1 or d["vk"] > args.maxvel:
        m_bh = d["m_bh"]
        continue
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

# XXX plot the observed mass
obs_data = load_obs_cluster_data()
for i, dat in enumerate(obs_data):
    ax.plot(dat[0], dat[1], c=r_col_mapper(dat[2]), ls="", marker="^", mec="k", mew=0.5)

ax.scatter(
    [], [], marker="^", c="gray", lw="0.5", ec="k", label=r"$\mathrm{LOS\,integrated}$"
)

# set dual y axis on second plot
SL.debug(f"BH mass is {m_bh:.2e} Msol")
ax.tick_params(axis="y", which="both", right=False)
axr = ax.secondary_yaxis("right", functions=(lambda x: x / m_bh, lambda x: x * m_bh))
axr.set_ylabel(r"$M/M_\bullet$")

# show core dispersion
xlim = ax.get_xlim()
ax.axvspan(
    xlim[0], core_dispersion, zorder=1, hatch="//", fc="none", ec="dimgray", lw=1
)
ax.text(
    0.1,
    0.7,
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
    2.5e6,
    r"$r_\mathrm{apo} > R_\mathrm{e}$",
)
ax.set_xlim(xlim)

ax.legend(fontsize="small")

bgs.plotting.savefig(figure_config.fig_path("bound.pdf"), force_ext=True)
