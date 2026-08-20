"""Quickly check SMBH binary parameters for several KETJU runs."""

import argparse
import os.path
from datetime import datetime
from typing import NamedTuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import dask
import baggins as bgs
import ketjugw

MYR = bgs.general.units.Myr
KPC = bgs.general.units.kpc


class LoadedBinary(NamedTuple):
    bh1: object
    bh2: object
    op: object


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quickly check SMBH binary parameters for several runs",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(type=str, help="path to directory", dest="path")
    parser.add_argument(
        "--timef",
        type=float,
        help="mask to times less than this (Myr)",
        default=-1,
        dest="tf",
    )
    parser.add_argument(
        "--time0",
        type=float,
        help="Initial time value (Myr)",
        default=0,
        dest="t0",
    )
    parser.add_argument(
        "-i",
        "--interp",
        action="store_true",
        dest="interp",
        help="interpolate BH data if needed",
    )
    parser.add_argument(
        "-s", "--save", action="store_true", dest="save", help="save figure"
    )
    parser.add_argument(
        "--publish", action="store_true", dest="publish", help="use publishing format"
    )
    parser.add_argument(
        "--orbits", action="store_true", dest="orbits", help="plot binary orbits"
    )
    parser.add_argument("-logt", action="store_true", dest="logt", help="log time axis")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        action="append",
        default=[],
        dest="extra_dirs",
        help="other directories to compare",
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
    return parser.parse_args()


@dask.delayed
def _load_binary_data(kf, SL, interp, t0, tf) -> Optional[LoadedBinary]:
    """Load one ketju file and compute its orbital parameters."""
    SL.debug(f"Reading: {kf}")
    try:
        bh1, bh2 = bgs.analysis.get_bound_binary(kf, interp=interp)
        merger_info = bgs.analysis.KetjuMergerInfo(kf)
        if merger_info.merged:
            SL.info(f"Merger in {kf}")
    except IndexError:
        SL.warning(f"No binaries found in: {kf} --> skipping...")
        return None

    t_myr = bh1.t / MYR
    time_mask = t_myr >= t0
    if tf >= 0:
        time_mask &= t_myr <= tf
    bh1 = bh1[time_mask]
    bh2 = bh2[time_mask]
    op = ketjugw.orbital_parameters(bh1, bh2)
    return LoadedBinary(bh1, bh2, op)


def _plot_orbit(ax2, bh1, bh2, color):
    """Plot relative position/velocity orbit for one binary."""
    pos = (bh1.x - bh2.x) / KPC
    vel = (bh1.v - bh2.v) / ketjugw.units.km_per_s
    for axidx, q in enumerate((pos, vel)):
        ax2[axidx, 0].plot(
            q[:, 0], q[:, 2], c=color, alpha=0.7, markevery=[-1], marker="o"
        )
        ax2[axidx, 1].plot(
            q[:, 0], q[:, 1], c=color, alpha=0.7, markevery=[-1], marker="o"
        )


def plot_directory(d, j, *, ax, ax2, args, SL, cols, linestyles, num_dirs, labels):
    """Load and plot every binary found under directory `d`. Returns count plotted."""
    kfs = bgs.utils.get_ketjubhs_in_dir(d)
    results = dask.compute(
        *(_load_binary_data(kf, SL, args.interp, args.t0, args.tf) for kf in kfs)
    )

    line_count = 0
    for i, (data, kf) in enumerate(zip(results, kfs)):
        if data is None:
            continue

        ls = linestyles[line_count // len(cols)]
        if num_dirs == 1:
            bgs.plotting.binary_param_plot(
                data.op,
                ax=ax,
                label=f"{kf.split('/')[-3]}",
                ls=ls,
            )
            color = cols[i % len(cols)]
        else:
            bgs.plotting.binary_param_plot(
                data.op,
                ax=ax,
                label=(labels[j] if i == 0 else ""),
                c=cols[j],
                alpha=0.6,
                markevery=1000,
                ls=ls,
            )
            color = cols[j % len(cols)]

        if args.orbits:
            _plot_orbit(ax2, data.bh1, data.bh2, color)
        line_count += 1
    return line_count


def main():
    args = parse_args()
    SL = bgs.setup_logger("script", args.verbosity)

    if args.publish:
        bgs.plotting.set_publishing_style()
        legend_kwargs = {"ncol": 2, "fontsize": "x-small"}
        fig_kwargs = {"transparent": True}
    else:
        legend_kwargs = {}
        fig_kwargs = {}

    ketju_dirs = [args.path, *args.extra_dirs]
    labels = None
    if args.extra_dirs:
        SL.debug(f"Directories are: {ketju_dirs}")
        labels = bgs.general.get_unique_path_part(ketju_dirs)
        SL.debug(f"Labels are: {labels}")

    fig2, ax2 = None, None
    if args.orbits:
        fig2, ax2 = plt.subplots(2, 2, sharex="row")
        for i, (s, u) in enumerate(zip(" v", ("kpc", "[km/s]"))):
            ax2[i, 0].set_xlabel(f"{s}x/{u}".lstrip())
            ax2[i, 0].set_ylabel(f"{s}z/{u}".lstrip())
            ax2[i, 1].set_xlabel(f"{s}x/{u}".lstrip())
            ax2[i, 1].set_ylabel(f"{s}y/{u}".lstrip())

    cols = bgs.plotting.mplColours()
    linestyles = bgs.plotting.mplLines()
    num_dirs = len(ketju_dirs)
    SL.debug(f"We will be plotting {num_dirs} different families...")

    ax = bgs.plotting.binary_param_plot(
        {"t": np.nan, "a_R": np.nan, "e_t": np.nan, "E": np.nan}, None
    )
    for axi in ax:
        axi.set_prop_cycle(None)

    total_sim_count = 0
    for j, d in enumerate(ketju_dirs):
        if not os.path.exists(d):
            SL.error(f"Path {d} does not exist!")
            raise FileNotFoundError(f"Path {d} does not exist!")

        line_count = plot_directory(
            d,
            j,
            ax=ax,
            ax2=ax2,
            args=args,
            SL=SL,
            cols=cols,
            linestyles=linestyles,
            num_dirs=num_dirs,
            labels=labels,
        )
        total_sim_count += line_count
        if line_count == 0:
            SL.warning(f"No bound BHs present in {d}")

    try:
        fig = ax[0].get_figure()
    except (IndexError, TypeError):
        SL.error("No bound BHs found!")
        return

    legend_defaults = {
        "loc": "best",
        "ncol": max(1, total_sim_count // 5),
        "columnspacing": 1,
    }
    ax[0].legend(**{**legend_defaults, **legend_kwargs})
    if args.logt:
        ax[0].set_xscale("log")
    if args.save:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        bgs.plotting.savefig(
            os.path.join(bgs.FIGDIR, f"merger/compare_binaries_{now}.png"),
            fig=fig,
            save_kwargs=fig_kwargs,
        )
        if args.orbits:
            bgs.plotting.savefig(
                os.path.join(bgs.FIGDIR, f"merger/compare_binaries_{now}_orbit.png"),
                fig=fig2,
                save_kwargs=fig_kwargs,
            )
    plt.show()


if __name__ == "__main__":
    main()
