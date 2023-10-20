import argparse
import os.path
import numpy as np
import scipy.optimize
import cm_functions as cmf
from helpers import stan_model_selector


parser = cmf.utils.argparse_for_stan("Run stan model for Core-Sersic model")
parser.add_argument("-m", "--model", help="model to run", type=str, choices=["simple", "hierarchy", "factor"], dest="model", default="hierarchy")
parser.add_argument("-c", "--compare", help="compare to naive statistics", action="store_true", dest="compare")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", console_level=args.verbose)

full_figsize = cmf.plotting.get_figure_size(args.publish, full=True, multiplier=[1.9, 1.9])

if args.type == "new":
    hmq_dir = args.dir
else:
    hmq_dir = None
SL.logger.debug(f"Input data read from {hmq_dir}")
analysis_params = cmf.utils.read_parameters(args.apf)

figname_base = f"hierarchical_models/density/"

if args.model == "simple":
    graham_model = stan_model_selector(
                    args, 
                    cmf.analysis.GrahamModelSimple, 
                    "stan/density/graham_simple.stan", 
                    "stan/density/graham_prior_simple.stan", 
                    figname_base, SL)
elif args.model == "factor":
    graham_model = stan_model_selector(
                    args,
                    cmf.analysis.GrahamModelKick,
                    "stan/density/graham_factor.stan",
                    "stan/density/graham_factor_prior_novk.stan",
                    figname_base, SL)
else:
    graham_model = stan_model_selector(
                    args, 
                    cmf.analysis.GrahamModelHierarchy, 
                    "stan/density/graham_hierarchy_0.stan", 
                    "stan/density/graham_prior_hierarchy_0.stan", 
                    figname_base, SL)


# load the observational data
graham_model.extract_data(analysis_params, hmq_dir)

SL.logger.info(f"Number of simulations with usable data: {graham_model.num_groups}")

if args.verbose == "DEBUG":
    graham_model.print_obs_summary()

# initialise the data dictionary
graham_model.set_stan_data()

if args.prior:
    # create the push-forward distribution for the prior model
    graham_model.sample_prior(sample_kwargs=analysis_params["stan"]["density_sample_kwargs"])

    # prior predictive check
    graham_model.all_prior_plots(full_figsize)
else:
    analysis_params["stan"]["density_sample_kwargs"]["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/density/{args.sample}/{graham_model.merger_id}")
    
    # run the model
    graham_model.sample_model(sample_kwargs=analysis_params["stan"]["density_sample_kwargs"])

    graham_model.determine_loo()

    ax = graham_model.all_posterior_plots(full_figsize)
    fig = ax[0].get_figure()

    if args.compare:
        # compare to naive estimates of latent parameter distributions
        all_optimal_pars = np.full((len(np.unique(graham_model.obs_collapsed["label"])), 6), np.nan)
        # note the argument order here is different to the function 
        # core_Sersic_profile
        log_core_sersic = lambda x, rb, Re, Ib, gamma, n, a: np.log10(cmf.literature.core_Sersic_profile(x, Re=Re, rb=rb, Ib=Ib, n=n, gamma=gamma, alpha=a))
        p_bounds = ([0, 0, 0, 0, 0, 0], [30, 30, np.inf, 20, 20, 30])
        for i in range(len(graham_model.obs["label"])):
            SL.logger.debug(f"Curve-fitting on sample {i}")
            x = graham_model.obs["R"][i]
            y = graham_model.obs["log10_proj_density_mean"][i]
            yerr = graham_model.obs["log10_proj_density_std"][i]
            popt, pcov = scipy.optimize.curve_fit(log_core_sersic, x, y, sigma=yerr, bounds=p_bounds, maxfev=2000*6)
            SL.logger.debug(f"Optimal parameters are: {popt}")
            all_optimal_pars[i,:] = popt
        all_optimal_pars[:,2] = np.log10(all_optimal_pars[:,2])
        SL.logger.debug(f"Least Squares optimal values:\n{all_optimal_pars}")
        naive_median, naive_spread = cmf.mathematics.quantiles_relative_to_median(all_optimal_pars, axis=0)
        for axi, nm, nsl, nsu in zip(ax, naive_median, naive_spread[0], naive_spread[1]):
            y = axi.get_ylim()[1]*1.1
            axi.errorbar(nm, y, xerr=np.atleast_2d([nsl, nsu]).T, c="k", capsize=2, fmt=".")
        cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"{graham_model.figname_base}_latentqty_compare.png"), fig=fig)
    
graham_model.print_parameter_percentiles(graham_model.latent_qtys)

