import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import figure_config


parser = argparse.ArgumentParser(description="Plot core-kick relation given a Stan sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-e", "--extract", help="extract data", action="store_true", dest="extract")
parser.add_argument("-n", "--number", help="number of drawn samples (with replacement)", dest="num", default=10000)
parser.add_argument("-p", "--parameter", help="parameter to plot", choices=["Re", "rb", "n", "a", "log10densb", "g"], default="rb", dest="param")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()


SL = cmf.setup_logger("script", args.verbosity)
data_file = "data/core-kick.pickle"
rng = np.random.default_rng()


if args.extract:
    main_path = "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs"
    analysis_params = cmf.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml")

    with os.scandir(main_path) as _it:
        subdirs = [entry.path for entry in _it if entry.is_dir() and "-v" in entry.name]
    subdirs.sort()
    for s in subdirs:
        SL.debug(f"Reading: {s}")

    figname_base = "ck"

    # put the data into a format we can pickle as numpy arrays for faster 
    # plotting
    data = {"rb":{}, "Re":{}, "n":{}, "log10densb":{}, "g":{}, "a":{}}

    # load the fits
    for subdir in subdirs:
        csv_files = cmf.utils.get_files_in_dir(subdir, ext=".csv")[:4]
        try:
            graham_model = cmf.analysis.GrahamModelHierarchy.load_fit(
                        model_file=os.path.join(
                        cmf.HOME,
                        "projects/collisionless-merger-sample/code/analysis_scripts/hierarchical_models/stan/density/graham_hierarchy.stan"
                    ),
                    fit_files=csv_files,
                    figname_base=figname_base
                )
        except:
            SL.warning(f"Unable to load data from directory: {subdir}. Skipping")
            continue
        SL.info(f"Loaded model from csv files {csv_files[0]}")

        graham_model.extract_data(analysis_params, None, binary=False)
        graham_model.set_stan_data()
        graham_model.sample_model(sample_kwargs=analysis_params["stan"]["density_sample_kwargs"])
        gid = graham_model.merger_id.split("-")[-1][1:]
        for k in data.keys():
            data[k][gid] = graham_model.sample_generated_quantity(f"{k}_posterior", state="OOS")
    cmf.utils.save_data(data, data_file)
else:
    data = cmf.utils.load_data(data_file)

fig, ax = plt.subplots(1,1)
# determine the ratio of rb / rb_initial
# TODO for now, use the v=0 data, but this should be changed to the parent run
# in the future
kick_vels = []
param = []
SL.warning(f"Determining distributions for parameter: {args.param}")
for k, v in data[args.param].items():
    if k == "__githash" or k == "__script": continue
    SL.info(f"Determining ratio for model {k}")
    kick_vels.append(float(k))

    normalisation = rng.choice(data[args.param]["0000"].flatten(), size=args.num) if args.param == "rb" else 1
    param.append(
        rng.choice(v.flatten(), size=args.num) / normalisation
    )
cmf.plotting.violinplot(param, kick_vels, ax, boxwidth=4, widths=60)

ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{kms}^{-1}$")
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

cmf.plotting.savefig(figure_config.fig_path(f"{args.param}-kick.pdf"), force_ext=True)
plt.show()