import argparse
import os.path
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Perform the eccentricity dispersion calculation in Nasim et al. 2020",
    allow_abbrev=False,
)
parser.add_argument(type=str, help="path to directory of HMQ files", dest="path")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument(
    "-s", "--save", action="store_true", dest="save", help="save figure"
)
parser.add_argument(
    "-sd",
    "--save-data",
    dest="save_data",
    type=str,
    default=None,
    help="directory to save data to",
)
parser.add_argument(
    "-P", "--Publish", action="store_true", dest="publish", help="use publishing format"
)
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    action="append",
    default=[],
    dest="extra_dirs",
    help="other directories of HMQ files to compare",
)
parser.add_argument(
    "-g",
    "--groups",
    choices=["e", "res"],
    help="Data groups",
    default="e",
    dest="groups",
)
parser.add_argument(
    "-c",
    "--cut",
    type=float,
    help="eccentricity cut in std.devs.",
    default=None,
    dest="ecut",
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

if args.publish:
    bgs.plotting.set_publishing_style()
    legend_kwargs = {"ncol": 2, "fontsize": "x-small"}
else:
    legend_kwargs = {}

hmq_dirs = []
hmq_dirs.append(args.path)
if args.extra_dirs:
    hmq_dirs.extend(args.extra_dirs)
    SL.debug(f"Directories are: {hmq_dirs}")

# read in the analysis parameters
analysis_params = bgs.utils.read_parameters(args.apf)

# list to store the values to plot
e_ini = []
mass_res = []
sigma_e = []
sigma_e_cut = []
suite_count = 0


for d in hmq_dirs:
    SL.debug(f"Reading from directory: {d}")
    # we can hack into the Kepler HM classes to extract the data
    HMQ_files = bgs.utils.get_files_in_dir(d)
    km = bgs.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(HMQ_files, analysis_params)
    eccs = np.array([np.nanmean(ecc) for ecc in km.obs["e"]])
    sigma_e.append(np.nanstd(eccs))
    if args.ecut is not None:
        sigma_e_cut.append(
            np.nanstd(
                eccs[np.abs(eccs - np.nanmean(eccs)) < args.ecut * np.nanstd(eccs)]
            )
        )
    else:
        sigma_e_cut.append(np.nan)
    try:
        assert np.allclose(
            np.diff(np.concatenate(km.obs["e_ini"])), np.zeros(km.num_groups - 1)
        )
    except AssertionError:
        SL.exception(
            "All simulations within a suite must have the same initial eccentricity!",
            exc_info=True,
        )
        raise
    e_ini.append(km.obs["e_ini"][0])
    min_bh_mass = min(min(km.obs["mass1"]), min(km.obs["mass2"]))
    try:
        assert np.allclose(
            np.diff(np.concatenate(km.obs["star_mass"])), np.zeros(km.num_groups - 1)
        )
    except AssertionError:
        SL.exception(
            f"Non-unique stellar masses! A fair comparison cannot be made. Stellar masses are {km.obs['star_mass']}",
            exc_info=True,
        )
        raise
    mass_res.append(min_bh_mass / km.obs["star_mass"][0])
    suite_count += 1

e_ini = np.concatenate(e_ini)
mass_res = np.concatenate(mass_res)

if args.groups == "e":
    try:
        assert np.allclose(np.diff(mass_res), np.zeros(suite_count - 1))
    except AssertionError:
        SL.exception(
            "Mass resolution must be constant when varying initial eccentricity!",
            exc_info=True,
        )
        raise
    x = e_ini
    fig_prefix = f"res-{mass_res[0]:.1e}"
else:
    try:
        assert np.allclose(np.diff(e_ini), np.zeros(suite_count - 1))
    except AssertionError:
        SL.exception(
            "Initial eccentricity must be constant when varying mass resolution!",
            exc_info=True,
        )
        raise
    x = mass_res
    fig_prefix = f"e0-{e_ini[0]:.3f}"

fig, ax = plt.subplots(1, 1)
ax.set_ylabel(r"$\sigma_e$")
ax.set_yscale("log")
sc = ax.scatter(x, sigma_e, lw=0.5, ec="k", label="Sims.", zorder=10)
if args.ecut is not None:
    ax.scatter(
        x,
        sigma_e_cut,
        lw=0.5,
        ec="k",
        label=f"Sims. ($<{args.ecut:.1f}\\sigma$)",
        zorder=10,
        c=sc.get_facecolor(),
        alpha=0.3,
        marker="s",
        s=sc.get_sizes() * 1.1,
    )

# add the Nasim line if plotting resolution on x axis
if args.groups == "res":
    ax.set_xlabel(r"$M_\bullet/m_\star$")
    ax.set_xscale("log")
    xseq = np.geomspace(0.9 * min(x), 1.1 * max(x))
    nasim_line_1 = lambda n, k: k / np.sqrt(n + k**2)
    nasim_line_2 = lambda n, k: k / np.sqrt(n**2 + k**2)
    for nl, lab, ls in zip(
        (nasim_line_1, nasim_line_2),
        (r"$\propto 1/\sqrt{N}$", r"$\propto 1/N$"),
        bgs.plotting.mplLines(),
    ):
        popt, pcov = scipy.optimize.curve_fit(nl, x, sigma_e)
        ax.plot(xseq, nl(xseq, *popt), c="k", label=lab, ls=ls)
    bgs.plotting.nice_log10_scale(ax, "x")
    ax.legend(fontsize="small")
else:
    ax.set_xlabel(r"$e_0$")
    bgs.plotting.nice_log10_scale(ax, "y")

if args.save:
    bgs.plotting.savefig(
        os.path.join(bgs.FIGDIR, f"merger/nasim_scatter_{fig_prefix}.png")
    )

if args.save_data is not None:
    data = dict(
        mass_res=mass_res,
        e_ini=e_ini,
        sigma_e=sigma_e,
        sigma_e_cut=sigma_e_cut,
        sigma_e_cut_threshold=args.ecut,
    )
    bgs.utils.save_data(
        data, os.path.join(args.save_data, f"nasim_scatter_{fig_prefix}.pickle")
    )

plt.show()
