import argparse
import os.path
import numpy as np
import h5py
from matplotlib import rcParams
import cm_functions as cmf


parser = cmf.utils.argparse_for_stan("Run Stan model for binary properties")
parser.add_argument("-m", "--model", help="model to run", type=str, choices=["simple", "hierarchy"], dest="model", default="hierarchy")
parser.add_argument("-t", "--thin", type=int, help="thin data", dest="thin", default=None)
args = parser.parse_args()

SL = cmf.setup_logger("script", console_level=args.verbose)

full_figsize = cmf.plotting.get_figure_size(args.publish, full=True)

if args.type == "new":
    hmq_dir = args.dir
else:
    hmq_dir = None
SL.debug(f"Input data read from {hmq_dir}")
analysis_params = cmf.utils.read_parameters(args.apf)

figname_base = f"hierarchical_models/binary/{args.sample}/"


if args.model == "simple":
    stan_model_file = "stan/binary/binary_simple.stan"
    if args.type == "loaded":
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "simple" in args.dir
        except AssertionError:
            SL.exception(f"Using model 'simple', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        kepler_model = cmf.analysis.KeplerModelSimple.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        kepler_model = cmf.analysis.KeplerModelSimple(model_file=stan_model_file, prior_file="stan/binary/binary_prior_simple.stan", figname_base=figname_base, num_OOS=args.NOOS)
else:
    stan_model_file = "stan/binary/binary_hierarchy_2.stan"
    if args.type == "loaded":
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "hierarchy" in args.load_file
        except AssertionError:
            SL.exception(f"Using model 'hierarchy', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        kepler_model = cmf.analysis.KeplerModelHierarchy.load_fit(model_file=stan_model_file, fit_files=args.dir, figname_base=figname_base)
    else:
        # sample
        kepler_model = cmf.analysis.KeplerModelHierarchy(model_file=stan_model_file, prior_file="stan/binary/binary_prior_hierarchy.stan", figname_base=figname_base, num_OOS=args.NOOS)

kepler_model.extract_data(analysis_params, hmq_dir)

SL.info(f"Number of simulations with usable data: {kepler_model.num_groups}")
try:
    assert kepler_model.num_groups >= analysis_params["stan"]["min_num_samples"]
except AssertionError:
    SL.exception(f'There are not enough groups to form a valid hierarchical model. Minimum number of groups is {analysis_params["stan"]["min_num_samples"]}, and we have {kepler_model.num_groups}!', exc_info=True)
    raise

# thin data
SL.debug(f"{kepler_model.num_obs} data points before thinning")
kepler_model.thin_observations(args.thin)
SL.debug(f"{kepler_model.num_obs} data points after thinning")

SL.debug(f"Median semimajor axis per group: {[np.nanmedian(g) for g in kepler_model.obs['a']]}")
SL.debug(f"Median eccentricity per group: {[np.nanmedian(g) for g in kepler_model.obs['e']]}")

if args.verbose == "DEBUG":
    kepler_model.print_obs_summary()

# initialise the data dictionary
kepler_model.set_stan_data()

if args.prior:
    # create the push-forward distribution for the prior model
    kepler_model.sample_prior(sample_kwargs=analysis_params["stan"]["binary_sample_kwargs"])

    # prior predictive checks
    kepler_model.all_prior_plots(full_figsize)

else:
    analysis_params["stan"]["binary_sample_kwargs"]["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/binary/{args.sample}/{merger_id}")

    # run the model
    kepler_model.sample_model(sample_kwargs=analysis_params["stan"]["binary_sample_kwargs"])

    kepler_model.determine_loo()

    ax = kepler_model.all_posterior_plots(full_figsize)
    kepler_model.all_posterior_OOS_plots(full_figsize)

kepler_model.print_parameter_percentiles(["a_hard", "e_hard"])
