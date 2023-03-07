import argparse
import os.path
import numpy as np
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Run Stan model for binary properties", allow_abbrev=False)
parser.add_argument(type=str, help="directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-m", "--model", help="model to run", type=str, choices=["simple", "hierarchy"], dest="model", default="hierarchy")
parser.add_argument("-p", "--prior", help="plot for prior", action="store_true", dest="prior")
parser.add_argument("-l", "--load", type=str, help="load previous stan file", dest="load_file", default=None)
parser.add_argument("-t", "--thin", type=int, help="thin data", dest="thin", default=10)
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", console_level=args.verbose)

if args.publish:
    cmf.plotting.set_publishing_style()
    full_figsize = rcParams["figure.figsize"]
    full_figsize[0] *= 2
else:
    full_figsize = None

HMQ_files = cmf.utils.get_files_in_dir(args.dir)
with h5py.File(HMQ_files[0], mode="r") as f:
    merger_id = f["/meta"].attrs["merger_id"]

figname_base = f"hierarchical_models/binary/{merger_id}/binary_properties-{merger_id}"

analysis_params = cmf.utils.read_parameters(args.apf)

if args.model == "simple":
    stan_model_file = "stan/binary_simple.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "simple" in args.load_file
        except AssertionError:
            SL.logger.exception(f"Using model 'simple', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        kepler_model = cmf.analysis.KeplerModelSimple.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        kepler_model = cmf.analysis.KeplerModelSimple(model_file=stan_model_file, prior_file="stan/binary_prior_simple.stan", figname_base=figname_base)
else:
    stan_model_file = "stan/binary_hierarchy.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "hierarchy" in args.load_file
        except AssertionError:
            SL.logger.exception(f"Using model 'hierarchy', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        kepler_model = cmf.analysis.KeplerModelHierarchy.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        kepler_model = cmf.analysis.KeplerModelHierarchy(model_file=stan_model_file, prior_file="stan/binary_prior_hierarchy.stan", figname_base=figname_base)

kepler_model.extract_data(HMQ_files, analysis_params)

SL.logger.info(f"Number of simulations with usable data: {kepler_model.num_groups}")
try:
    assert kepler_model.num_groups >= analysis_params["stan"]["min_num_samples"]
except AssertionError:
    SL.logger.exception(f'There are not enough groups to form a valid hierarchical model. Minimum number of groups is {analysis_params["stan"]["min_num_samples"]}, and we have {kepler_model.num_groups}!', exc_info=True)
    raise

# thin data
SL.logger.debug(f"{kepler_model.num_obs} data points before thinning")
kepler_model.thin_observations(args.thin)
SL.logger.debug(f"{kepler_model.num_obs} data points after thinning")

SL.logger.debug(f"Median semimajor axis per group: {[np.nanmedian(g) for g in kepler_model.obs['a']]}")
SL.logger.debug(f"Median eccentricity per group: {[np.nanmedian(g) for g in kepler_model.obs['e']]}")

if args.verbose == "DEBUG":
    kepler_model.print_obs_summary()

# initialise the data dictionary
# note that we assume initial normalised pericentre distance is part of the 
# merger ID
stan_data = dict(
    N_groups = len(kepler_model.obs["label"]),
    e_0 = cmf.initialise.e_from_rperi(float(merger_id.split("-")[-1]))
)

if args.prior:
    # create the push-forward distribution for the prior model
    kepler_model.sample_prior(data=stan_data, sample_kwargs=analysis_params["stan"]["binary_sample_kwargs"])

    # prior predictive checks
    kepler_model.all_prior_plots(full_figsize)

else:
    stan_data.update(dict(
        N_tot = len(kepler_model.obs_collapsed["log10_angmom_corr_red"]),
        group_id = kepler_model.obs_collapsed["label"],
        log10_angmom = kepler_model.obs_collapsed["log10_angmom_corr_red"]
    ))

    analysis_params["stan"]["binary_sample_kwargs"]["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/binary/{merger_id}")

    # run the model
    kepler_model.sample_model(data=stan_data, sample_kwargs=analysis_params["stan"]["binary_sample_kwargs"])

    kepler_model.determine_loo()

    ax = kepler_model.all_posterior_plots(full_figsize)

kepler_model.print_parameter_percentiles(["a_hard", "e_hard"])

