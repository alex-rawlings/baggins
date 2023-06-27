import argparse
import os.path
import numpy as np
import h5py
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Run Stan model for Quinlan hardening parameter", allow_abbrev=False)
parser.add_argument(type=str, help="directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-m", "--model", help="model to run", type=str, choices=["simple", "hierarchy"], dest="model", default="hierarchy")
parser.add_argument("-p", "--prior", help="plot for prior", action="store_true", dest="prior")
parser.add_argument("-l", "--load", type=str, help="load previous stan file", dest="load_file", default=None)
parser.add_argument("-t", "--thin", type=int, help="thin data", dest="thin", default=10)
parser.add_argument("-s", "--sample", help="sample set", type=str, dest="sample", choices=["mcs", "perturb"], default="mcs")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", console_level=args.verbose)

full_figsize = cmf.plotting.get_figure_size(args.publish, full=True)

HMQ_files = cmf.utils.get_files_in_dir(args.dir)
with h5py.File(HMQ_files[0], mode="r") as f:
    merger_id = f["/meta"].attrs["merger_id"]
    e_ini = cmf.initialise.e_from_rperi(float(merger_id.split("-")[-1]))
    if args.sample:
        merger_id = "-".join(merger_id.split("-")[:2])

figname_base = f"hierarchical_models/hardening/{args.sample}/{merger_id}/quinlan_hardening-{merger_id}"

analysis_params = cmf.utils.read_parameters(args.apf)

if args.model == "simple":
    stan_model_file = "stan/hardening/quinlan_simple.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "simple" in args.load_file
        except AssertionError:
            SL.logger.exception(f"Using model 'simple', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        quinlan_model = cmf.analysis.QuinlanModelSimple.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        quinlan_model = cmf.analysis.QuinlanModelSimple(model_file=stan_model_file, prior_file="stan/hardening/quinlan_simple_prior.stan", figname_base=figname_base)
else:
    stan_model_file = "stan/hardening/quinlan_hierarchy.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "hierarchy" in args.load_file
        except AssertionError:
            SL.logger.exception(f"Using model 'hierarchy', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        quinlan_model = cmf.analysis.QuinlanModelHierarchy.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        quinlan_model = cmf.analysis.QuinlanModelHierarchy(model_file=stan_model_file, prior_file="stan/hardening/quinlan_hierarchy_prior.stan", figname_base=figname_base)

quinlan_model.extract_data(HMQ_files, analysis_params)
if "e_ini" in quinlan_model.obs:
    e_ini = None
else:
    SL.logger.warning(f"Initial eccentricity has been determined from the file name, and has value: {e_ini:.3f}")

SL.logger.info(f"Number of simulations with usable data: {quinlan_model.num_groups}")

# thin data
SL.logger.debug(f"{quinlan_model.num_obs} data points before thinning")
quinlan_model.thin_observations(args.thin)
SL.logger.debug(f"{quinlan_model.num_obs} data points after thinning")

# initialise the data dictionary
quinlan_model.set_stan_dict(e_ini)

if args.prior:
    # create the push-forward distribution for the prior model
    quinlan_model.sample_prior(sample_kwargs=analysis_params["stan"]["hardening_sample_kwargs"])

    # prior predictive checks
    quinlan_model.all_prior_plots(full_figsize)
else:
    analysis_params["stan"]["hardening_sample_kwargs"]["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/hardening/{args.sample}/{merger_id}")

    # run the model
    quinlan_model.sample_model(sample_kwargs=analysis_params["stan"]["hardening_sample_kwargs"])

    quinlan_model.determine_loo()

    ax = quinlan_model.all_posterior_plots(full_figsize)

quinlan_model.print_parameter_percentiles(quinlan_model.latent_qtys)

