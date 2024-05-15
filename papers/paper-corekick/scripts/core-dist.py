import argparse
import os.path
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use

    use("Agg")
    import matplotlib.pyplot as plt
import baggins as bgs
import figure_config  # noqa

parser = argparse.ArgumentParser(
    "Determine core - kick relation",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    type=str, help="new sample or load previous", choices=["new", "loaded"], dest="type"
)
parser.add_argument(
    "-N", "--NumSamples", type=int, help="number OOS values", dest="NOOS", default=1000
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbosity",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)

bgs.plotting.check_backend()

# set the stan model file
stan_file = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/gaussian_processes/stan/gp_analytic.stan"

# load necessary data
# data from previous core fitting routine
datafile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
# simulation output data at the moment just before merger
ketju_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output"
# load the fit files
fit_files = "/scratch/pjohanss/arawling/collisionless_merger/stan_files/gp-core-kick-relation/gp_analytic-20240513134707_*.csv"
# set the escape velocity in km/s
ESCAPE_VEL = 1800
figname_base = "core-study/gp-rb-dist"
rng = np.random.default_rng(99918082)

if args.type == "new":
    ck = bgs.analysis.VkickCoreradiusGP(
        stan_file,
        "",
        figname_base=figname_base,
        escape_vel=ESCAPE_VEL,
        premerger_ketjufile=ketju_file,
        rng=rng,
    )

else:
    ck = bgs.analysis.VkickCoreradiusGP.load_fit(
        stan_file,
        fit_files,
        figname_base,
        escape_vel=ESCAPE_VEL,
        premerger_ketjufile=ketju_file,
        rng=rng,
    )

ck.extract_data(d=datafile)

if args.verbosity == "DEBUG":
    ck.print_obs_summary()
ck.set_stan_data()

sample_kwargs = {
    "output_dir": os.path.join(bgs.DATADIR, "stan_files/gp-core-kick-relation"),
    "adapt_delta": 0.99,
    "max_treedepth": 15,
}
ck.sample_model(sample_kwargs=sample_kwargs)

if args.verbosity == "DEBUG":
    plt.hist(ck.stan_data["x2"], 50, density=True)
    plt.xlabel(r"$v_\mathrm{kick}$")
    plt.ylabel("PDF")
    plt.show()

ck.all_plots()
ck.print_parameter_percentiles(ck.latent_qtys)
