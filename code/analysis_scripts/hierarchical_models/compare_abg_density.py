import argparse
import os.path
import yaml
import matplotlib.pyplot as plt
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Compare ABG density profile parameters",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(dest="input", help="yml file with runs to compare", type=str)
parser.add_argument(dest="save", help="save figure to", type=str)
parser.add_argument(
    "-p",
    "--param",
    dest="param",
    type=str,
    choices=["a", "b", "g", "rS", "rhoS", "err", "all", "OOS"],
    default="all",
    help="parameter to compare",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

if args.param == "all":
    fig, ax = plt.subplots(3, 2)
else:
    fig, ax = plt.subplots()

with open(args.input, "r") as f:
    stan_files = yaml.safe_load(f)


def data_generator():
    # get the simulation time from file names
    # TODO assumes a specific format, not robust
    for fam in stan_files.values():
        for run_id, sfiles in fam.items():
            yield float(run_id[-3:]), run_id, sfiles["data"], sfiles["stan"]


ts = [t[0] for t in data_generator()]
cmin = min(ts)
SL.debug(f"Minimum colour corresponds to {cmin}")
cmax = max(ts)
SL.debug(f"Minimum colour corresponds to {cmax}")
cmapper, sm = bgs.plotting.create_normed_colours(cmin, cmax, cmap="flare")

for t, run_id, dfile, sfile in data_generator():
    abg = bgs.analysis.ABGDensityModelSimple.load_fit(sfile, figname_base="abg_density")
    SL.debug(f"Reading data from {dfile}")
    abg.read_data_from_txt(dfile, mergerid=run_id, skiprows=1)
    abg.set_stan_data()
    abg.sample_model(diagnose=False)
    if args.param == "all":
        abg.plot_generated_quantity_dist(
            abg.latent_qtys,
            state="OOS",
            xlabels=abg.latent_qtys_labs,
            ax=ax,
            color=cmapper(t),
            save=False,
            plot_kwargs={"lw": 2},
        )
    elif args.param == "OOS":
        abg.posterior_OOS_specific_hdi_plot(
            xmodel="r_OOS",
            ymodel=abg.folded_qtys_posterior[0],
            ax=ax,
            hdi=50,
            plot_kwargs={"c": cmapper(t)},
            label=f"{t:.1f} Gyr",
        )
    else:
        raise NotImplementedError

if args.param == "all":
    plt.colorbar(sm, location="top", ax=ax[0, :])
elif args.param == "OOS":
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r/\mathrm{kpc}$")
    ax.set_ylabel(r"$\rho/(\mathrm{M}_\odot\,\mathrm{kpc}^{-3})$")
    ax.legend(title="50% HDI")

bgs.plotting.savefig(os.path.join(args.save, f"ABG_comp_{args.param}.png"), fig=fig)
