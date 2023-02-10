import argparse
from datetime import datetime
import os.path
import numpy as np
import scipy.optimize
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Run stan model for Core-Sersic model.", allow_abbrev=False)
parser.add_argument(type=str, help="Directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-m", "--model", help="model to run", type=str, choices=["simple", "hierarchy"], dest="model", default="hierarchy")
parser.add_argument("-p", "--prior", help="Plot for prior", action="store_true", dest="prior")
parser.add_argument("-l", "--load", type=str, help="Load previous stan file", dest="load_file", default=None)
parser.add_argument("-r", "--r", type=int, dest="random_sample", default=None, help="randomly sample x observations from each population member")
parser.add_argument("-c", "--compare", help="Compare to naive statistics", action="store_true", dest="compare")
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

figname_base = f"hierarchical_models/density/{merger_id}/graham_density-{merger_id}"

analysis_params = cmf.utils.read_parameters(args.apf)

if args.model == "simple":
    stan_model_file = "stan/graham_simple.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "simple" in args.load_file
        except AssertionError:
            SL.logger.exception(f"Using model 'simple', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        graham_model = cmf.analysis.GrahamModelSimple.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        graham_model = cmf.analysis.GrahamModelSimple(model_file=stan_model_file, prior_file="stan/graham_prior_simple.stan", figname_base=figname_base)
else:
    stan_model_file = "stan/graham_hierarchy.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert "hierarchy" in args.load_file
        except AssertionError:
            SL.logger.exception(f"Using model 'hierarchy', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        graham_model = cmf.analysis.GrahamModelSimple.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
    else:
        # sample
        graham_model = cmf.analysis.GrahamModelSimple(model_file=stan_model_file, prior_file="stan/graham_prior_hierarchy.stan", figname_base=figname_base)

# load the observational data
graham_model.extract_data(HMQ_files, analysis_params)

# maybe randomly select some data
if args.random_sample is not None:
    graham_model.random_obs_select(args.random_sample, "name")

SL.logger.info(f"Number of simulations with usable data: {graham_model.num_groups}")
assert graham_model.num_groups >= analysis_params["stan"]["min_num_samples"]

# initialise the data dictionary
stan_data = {}

if args.prior:
    # create the push-forward distribution for the prior model
    R_unique = np.unique(graham_model.obs["R"])
    stan_data.update(dict(
        R = R_unique,
        N_tot = len(R_unique)
    ))

    graham_model.sample_prior(data=stan_data, sample_kwargs=analysis_params["stan"]["sample_kwargs"])

    # prior predictive check
    graham_model.all_prior_plots(full_figsize)
else:
    # create the push-forward distribution for the posterior model
    stan_data.update(dict(
                N_tot = graham_model.obs_len,
                R = graham_model.obs_collapsed["R"],
                N_child = len(np.unique(graham_model.obs_collapsed["label"])),
                child_id = graham_model.obs_collapsed["label"],
                log10_surf_rho = graham_model.obs_collapsed["log10_proj_density_mean"],
                log10_surf_rho_err = graham_model.obs_collapsed["log10_proj_density_std"],
    ))

    if args.random_sample:
        now = datetime.now()
        now = now.strftime("%Y%m%d-%H%M%S")
        analysis_params["stan"]["sample_kwargs"]["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/{merger_id}-{now}")
    else:
        analysis_params["stan"]["sample_kwargs"]["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/{merger_id}")
    # run the model
    graham_model.sample_model(data=stan_data, sample_kwargs=analysis_params["stan"]["sample_kwargs"])

    graham_model.determine_loo("log10_surf_rho_posterior")

    ax = graham_model.all_posterior_plots(full_figsize)
    fig = plt.gcf()

    if args.compare:
        # compare to naive estimates of latent parameter distributions
        all_optimal_pars = np.full((len(np.unique(graham_model.categorical_label)), 5), np.nan)
        # note the argument order here is different to the function 
        # core_Sersic_profile
        log_core_sersic = lambda x, rb, Re, Ib, gamma, n, a: np.log10(cmf.literature.core_Sersic_profile(x, Re=Re, rb=rb, Ib=Ib, n=n, gamma=gamma, alpha=a))
        p_bounds = ([0, 0, 0, 0, 0, 0], [30, 30, np.inf, 20, 20, 30])
        for i, n in enumerate(set(graham_model.categorical_label)):
            SL.logger.debug(f"Curve-fitting on sample {n}")
            mask = graham_model.categorical_label == n
            x = graham_model.obs["R"][mask]
            y = graham_model.obs["log10_proj_density_mean"][mask]
            yerr = graham_model.obs["log10_proj_density_std"][mask]
            popt, pcov = scipy.optimize.curve_fit(log_core_sersic, x, y, sigma=yerr, bounds=p_bounds, maxfev=2000*6)
            SL.logger.debug(f"Optimal parameters are: {popt}")
            all_optimal_pars[i,:] = popt
        all_optimal_pars[:,2] /= 1e9
        SL.logger.debug(f"Least Squares optimal values:\n{all_optimal_pars}")
        naive_median, naive_spread = cmf.mathematics.quantiles_relative_to_median(all_optimal_pars, axis=0)
        for axi, nm, ns in zip(ax, naive_median, naive_spread):
            y = axi.get_ylim()[1]*1.1
            axi.errorbar(nm, y, xerr=ns[::,np.newaxis], c="k", capsize=2, fmt=".")
        cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"{figname_base}_latentqty_compare.png"), fig=fig)
    
    graham_model.print_parameter_percentiles(graham_model.latent_qtys)

#plt.show()


