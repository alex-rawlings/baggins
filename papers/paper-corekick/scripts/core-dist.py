import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import baggins as bgs
import figure_config

parser = argparse.ArgumentParser(
    "Determine core - kick relation",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    type=str,
    help="new sample or load previous",
    choices=["new", "loaded", "comp"],
    dest="type",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="model type",
    choices=["exp", "lin", "sigmoid"],
    dest="model",
    default="exp",
)
parser.add_argument(
    "-n", "--nodiag", action="store_false", help="prevent Stan diagnosis", dest="diag"
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

# load necessary data
# data from previous core fitting routine
datafile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
# simulation output data at the moment just before merger
ketju_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output"
# set the stan model file base path
stan_file_base = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/core_kick_relation"
# set the escape velocity in km/s
ESCAPE_VEL = 1800
rng = np.random.default_rng(42)

if args.type == "new" or args.type == "loaded":
    # usage 1: run fit (either new or loaded) for a given model
    # set the stan model file
    stan_file = os.path.join(stan_file_base, f"core-kick-{args.model}.stan")
    figname_base = f"core-study/rb-kick-models/{args.model}/{args.model}"
    stan_output = f"stan_files/core-kick-relation/{args.model}-model"
    if args.type == "new":
        stan_new_kwargs = {
            "model_file": stan_file,
            "prior_file": "",
            "figname_base": figname_base,
            "escape_vel": ESCAPE_VEL,
            "premerger_ketjufile": ketju_file,
            "rng": rng,
        }
        if args.model == "exp":
            ck = bgs.analysis.CoreKickExp(**stan_new_kwargs)
        elif args.model == "lin":
            ck = bgs.analysis.CoreKickLinear(**stan_new_kwargs)
        else:
            ck = bgs.analysis.CoreKickSigmoid(**stan_new_kwargs)
    else:
        csv_files = bgs.utils.get_files_in_dir(
            os.path.join(bgs.DATADIR, stan_output), ext=".csv"
        )[-4:]
        stan_load_kwargs = {
            "model_file": stan_file,
            "fit_files": csv_files,
            "figname_base": figname_base,
            "escape_vel": ESCAPE_VEL,
            "premerger_ketjufile": ketju_file,
            "rng": rng,
        }
        if args.model == "exp":
            ck = bgs.analysis.CoreKickExp.load_fit(**stan_load_kwargs)
        elif args.model == "lin":
            ck = bgs.analysis.CoreKickLinear.load_fit(**stan_load_kwargs)
        else:
            ck = bgs.analysis.CoreKickSigmoid.load_fit(**stan_load_kwargs)

    ck.extract_data(d=datafile)

    if args.verbosity == "DEBUG":
        ck.print_obs_summary()
    ck.set_stan_data()

    sample_kwargs = {
        "output_dir": os.path.join(bgs.DATADIR, stan_output),
        "adapt_delta": 0.99,
        "max_treedepth": 15,
    }
    ck.sample_model(sample_kwargs=sample_kwargs, diagnose=args.diag)

    if args.verbosity == "DEBUG":
        plt.hist(ck.stan_data["x2"], 50, density=True)
        plt.xlabel(r"$v_\mathrm{kick}$")
        plt.ylabel("PDF")
        plt.show()

    ck.determine_loo()
    ck.all_plots()
    ck.print_parameter_percentiles(ck.latent_qtys)
else:
    # usage 2: run model comparison of most recently fit models
    fig, ax = plt.subplots()
    stan_load_kwargs = {
        "figname_base": "core-study/rb-kick-models/comparison/comp",
        "escape_vel": ESCAPE_VEL,
        "premerger_ketjufile": ketju_file,
        "rng": rng,
    }
    models = [
        bgs.analysis.CoreKickExp.load_fit(
            model_file=os.path.join(stan_file_base, "core-kick-exp.stan"),
            fit_files=bgs.utils.get_files_in_dir(
                os.path.join(bgs.DATADIR, "stan_files/core-kick-relation/exp-model"),
                ext=".csv",
            )[-4:],
            **stan_load_kwargs,
        ),
        bgs.analysis.CoreKickLinear.load_fit(
            model_file=os.path.join(stan_file_base, "core-kick-lin.stan"),
            fit_files=bgs.utils.get_files_in_dir(
                os.path.join(bgs.DATADIR, "stan_files/core-kick-relation/lin-model"),
                ext=".csv",
            )[-4:],
            **stan_load_kwargs,
        ),
        bgs.analysis.CoreKickSigmoid.load_fit(
            model_file=os.path.join(stan_file_base, "core-kick-sigmoid.stan"),
            fit_files=bgs.utils.get_files_in_dir(
                os.path.join(
                    bgs.DATADIR, "stan_files/core-kick-relation/sigmoid-model"
                ),
                ext=".csv",
            )[-4:],
            **stan_load_kwargs,
        ),
    ]
    loo_dict = {"Exponential": None, "Linear": None, "Sigmoid": None}
    for n, m, c in zip(
        loo_dict.keys(), models, figure_config.custom_colors_shuffled[1:]
    ):
        SL.info(f"Doing model: {n}")
        m.extract_data(d=datafile)
        m.set_stan_data()
        m.sample_model(diagnose=False)
        loo_dict[n] = m.determine_loo()
        # plot rb distribution for different models, without kick velocity 
        # restriction
        m.plot_generated_quantity_dist(
            ["rb_posterior"],
            state="OOS",
            xlabels=m._folded_qtys_labs,
            save=False,
            ax=ax,
            color=c,
            plot_kwargs={"ls":"--", "alpha":0.4},
        )

        # plot rb distribution for different models, with kick velocity 
        # restriction
        m.set_stan_data_OOS(restrict_v=True)
        m.sample_generated_quantity("rb_posterior", force_resample=True, state="OOS")
        m.plot_generated_quantity_dist(
            ["rb_posterior"],
            state="OOS",
            xlabels=m._folded_qtys_labs,
            save=False,
            ax=ax,
            label=n,
            color=c
        )

        for lq in m.latent_qtys:
            hdi = az.hdi(m.sample_generated_qty(lq))
            SL.info(f"1-sigma (68%) HDI for {lq} is {hdi}")


    ax.legend()
    # set xlimits by hand
    ax.set_xlim(0, 8)
    # add a secondary axis, turning off ticks from the top axis (if they are there)
    ax.tick_params(axis="x", which="both", top=False)
    rb02kpc = lambda x: x * models[-1].rb0
    kpc2rb0 = lambda x: x / models[-1].rb0
    secax = ax.secondary_xaxis("top", functions=(rb02kpc, kpc2rb0))
    secax.set_xlabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
    bgs.plotting.savefig(figure_config.fig_path("rb_pdf.pdf"), fig=fig, force_ext=True)
    comp = az.compare(loo_dict, ic="loo")
    print("Model comparison")
    print(comp)
