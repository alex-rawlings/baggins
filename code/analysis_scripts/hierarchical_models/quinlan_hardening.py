import os.path
import baggins as bgs
from .helpers import stan_model_selector


parser = bgs.utils.argparse_for_stan("Run Stan model for Quinlan hardening model")
parser.add_argument(
    "-m",
    "--model",
    help="model to run",
    type=str,
    choices=["simple", "hierarchy"],
    dest="model",
    default="hierarchy",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

full_figsize = bgs.plotting.get_figure_size(
    args.publish, full=True, multiplier=[1.9, 1.9]
)

if args.type == "new":
    hmq_dir = args.dir
else:
    hmq_dir = None
SL.debug(f"Input data read from {hmq_dir}")
analysis_params = bgs.utils.read_parameters(args.apf)

figname_base = f"hierarchical_models/hardening/{args.sample}/"

if args.model == "simple":
    quinlan_model = stan_model_selector(
        args,
        bgs.analysis.QuinlanModelSimple,
        "stan/hardening/quinlan_simple.stan",
        "stan/hardening/quinlan_simple_prior.stan",
        figname_base,
        SL,
    )
else:
    quinlan_model = stan_model_selector(
        args,
        bgs.analysis.QuinlanModelHierarchy,
        "stan/hardening/quinlan_hierarchy.stan",
        "stan/hardening/quinlan_hierarchy_prior.stan",
        figname_base,
        SL,
    )

quinlan_model.extract_data(analysis_params, hmq_dir)

SL.info(f"Number of simulations with usable data: {quinlan_model.num_groups}")

# thin data
SL.debug(f"{quinlan_model.num_obs} data points before thinning")
quinlan_model.thin_observations(analysis_params["stan"]["thin_data"])
SL.debug(f"{quinlan_model.num_obs} data points after thinning")

# initialise the data dictionary
quinlan_model.set_stan_data()

if args.prior:
    # create the push-forward distribution for the prior model
    quinlan_model.sample_prior(
        sample_kwargs=analysis_params["stan"]["hardening_sample_kwargs"]
    )

    # prior predictive checks
    quinlan_model.all_prior_plots(full_figsize)
else:
    analysis_params["stan"]["hardening_sample_kwargs"]["output_dir"] = os.path.join(
        bgs.DATADIR, f"stan_files/hardening/{args.sample}/{quinlan_model.merger_id}"
    )

    # run the model
    quinlan_model.sample_model(
        sample_kwargs=analysis_params["stan"]["hardening_sample_kwargs"]
    )

    quinlan_model.determine_loo()

    ax = quinlan_model.all_posterior_pred_plots(full_figsize)
    quinlan_model.all_posterior_OOS_plots(full_figsize)
quinlan_model.plot_merger_timescale()

quinlan_model.print_parameter_percentiles(quinlan_model.latent_qtys)
