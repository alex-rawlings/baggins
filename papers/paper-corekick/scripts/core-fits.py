import argparse
import os
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import figure_config
import arviz as az


parser = argparse.ArgumentParser(
    description="Plot core fits given a Stan sample",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-n",
    "--number",
    help="number of drawn samples (with replacement)",
    dest="num",
    default=10000,
)
parser.add_argument(
    "-p",
    "--parameter",
    help="parameter to plot",
    choices=["Re", "rb", "n", "a", "log10densb", "g", "all", "OOS"],
    default="rb",
    dest="param",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=cmf.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()


SL = cmf.setup_logger("script", args.verbosity)
data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
rng = np.random.default_rng(42)
col_list = figure_config.color_cycle_shuffled.by_key()["color"]


if args.extract:
    main_path = "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs"
    analysis_params = cmf.utils.read_parameters(
        "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"
    )

    with os.scandir(main_path) as _it:
        subdirs = [entry.path for entry in _it if entry.is_dir() and "-v" in entry.name]
    subdirs.sort()
    for s in subdirs:
        SL.debug(f"Reading: {s}")

    figname_base = "ck"

    # put the data into a format we can pickle as numpy arrays for faster
    # plotting
    data = {
        "rb": {},
        "Re": {},
        "n": {},
        "log10densb": {},
        "g": {},
        "a": {},
        "R_OOS": {},
        "log10_surf_rho": {},
    }

    # load the fits
    for subdir in subdirs:
        csv_files = cmf.utils.get_files_in_dir(subdir, ext=".csv")[:4]
        try:
            graham_model = cmf.analysis.GrahamModelHierarchy.load_fit(
                model_file=os.path.join(
                    cmf.HOME,
                    "projects/collisionless-merger-sample/code/analysis_scripts/hierarchical_models/stan/density/graham_hierarchy.stan",
                ),
                fit_files=csv_files,
                figname_base=figname_base,
            )
        except:
            SL.warning(f"Unable to load data from directory: {subdir}. Skipping")
            continue
        SL.info(f"Loaded model from csv files {csv_files[0]}")

        graham_model.extract_data(analysis_params, None, binary=False)
        graham_model.set_stan_data()
        graham_model.sample_model(
            sample_kwargs=analysis_params["stan"]["density_sample_kwargs"]
        )
        gid = graham_model.merger_id.split("-")[-1][1:]
        for k in data.keys():
            if k == "R_OOS":
                data[k][gid] = graham_model.stan_data[k]
            else:
                data[k][gid] = graham_model.sample_generated_quantity(
                    f"{k}_posterior", state="OOS"
                )
    cmf.utils.save_data(data, data_file)
else:
    data = cmf.utils.load_data(data_file)


def _helper(param_name, ax):
    kick_vels = []
    param = []
    SL.warning(f"Determining distributions for parameter: {param_name}")
    for k, v in data[param_name].items():
        if k == "__githash" or k == "__script":
            continue
        if float(k) > 900:
            break
        SL.info(f"Determining ratio for model {k}")
        kick_vels.append(float(k))
        v = v[~np.isnan(v)]
        # determine the ratio of rb / rb_initial
        normalisation = (
            rng.choice(data[param_name]["0000"].flatten(), size=args.num)
            if param_name == "rb"
            else 1
        )
        param.append(rng.choice(v.flatten(), size=args.num) / normalisation)
    bp = ax.boxplot(
        param,
        positions=kick_vels,
        showfliers=False,
        widths=40,
        manage_ticks=False,
        patch_artist=True,
    )
    for p in bp["boxes"]:
        p.set_facecolor(col_list[0])
        p.set_edgecolor(p.get_facecolor())
        p.set_alpha(0.3)
    for m in bp["medians"]:
        m.set_color(p.get_facecolor())
        m.set_linewidth(2)
        m.set_alpha(1)
    for w in chain(bp["whiskers"], bp["caps"]):
        w.set_color("#373737")


xlabel = r"$v_\mathrm{kick}/\mathrm{kms}^{-1}$"
if args.param == "all":
    ylabs = dict(
        Re=r"$R_\mathrm{e}/\mathrm{kpc}$",
        rb=r"$r_\mathrm{b}/r_\mathrm{b,0}$",
        n=r"$n$",
        a=r"$\alpha$",
        log10densb=r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
        g=r"$\gamma$",
    )
    fig, ax = plt.subplots(2, 3, sharex="all")
    fig.set_figwidth(2 * fig.get_figwidth())
    fig.set_figheight(1.2 * fig.get_figheight())
    for axi in ax[-1, :]:
        axi.set_xlabel(xlabel)
    for i, pname in enumerate(("rb", "log10densb", "g", "Re", "a", "n")):
        SL.info(f"Making plot for {pname}")
        _helper(pname, ax.flat[i])
        ax.flat[i].set_ylabel(ylabs[pname])
    fname = f"{args.param}-kick.pdf"
elif args.param == "OOS":
    fig, ax = plt.subplots(1, 1)
    axins = ax.inset_axes(
        [0.07, 0.02, 0.6, 0.5],
        xlim=(-0.95, 0),
        ylim=(9.25, 9.85),
        xticklabels=[],
        yticklabels=[],
        xticks=[],
        yticks=[],
    )

    def cols():
        for c in figure_config.custom_colors_shuffled:
            yield c

    cgen = cols()
    for k, v in data["R_OOS"].items():
        if k == "__githash" or k == "__script":
            continue
        if k not in ("0000", "0240", "0480", "0720", "0900"):
            continue
        SL.info(f"Determining density for model {k}")
        c = next(cgen)

        def _dens_plotter(axi):
            az.plot_hdi(
                np.log10(v),
                data["log10_surf_rho"][k],
                hdi_prob=0.25,
                ax=axi,
                smooth=True,
                hdi_kwargs={"skipna": True},
                fill_kwargs={
                    "label": f"{float(k):.1f}",
                    "color": c,
                    "edgecolor": c,
                    "lw": 0.5,
                },
            )

        _dens_plotter(ax)
        _dens_plotter(axins)
    ax.indicate_inset_zoom(axins, edgecolor="k")
    ax.set_xlabel(r"$\log_{10}(R/\mathrm{kpc})$")
    ax.set_ylabel(
        r"$\log_{10}\left(\Sigma(R)/\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)$"
    )
    ax.legend(
        title=r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$", loc="upper right"
    )
    fname = "density.pdf"
else:
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(xlabel)
    _helper(args.param, ax)
    if args.param == "rb":
        ax.set_ylabel(r"$r_\mathrm{b}/r_{\mathrm{b},0}$")
    elif args.param == "Re":
        ax.set_ylabel(r"$R_\mathrm{e}/\mathrm{kpc}$")
    elif args.param == "n":
        ax.set_ylabel(r"$n$")
    elif args.param == "a":
        ax.set_ylabel(r"$\alpha$")
    elif args.param == "g":
        ax.set_ylabel(r"$\gamma$")
    else:
        ax.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
    fname = f"{args.param}-kick.pdf"

cmf.plotting.savefig(figure_config.fig_path(fname), force_ext=True)
plt.show()
