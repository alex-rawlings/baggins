import argparse
from datetime import datetime
import os.path
import numpy as np
import scipy.optimize
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Run stan model for Core-Sersic model.", allow_abbrev=False)
parser.add_argument(type=str, help="Directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
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
stan_model_file = "stan/graham_new_3a.stan"

if args.load_file is not None:
    # load a previous sample for improved performance: no need to resample the
    # likelihood function
    graham_model = cmf.analysis.StanModel_2D.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
else:
    # sample
    graham_model = cmf.analysis.StanModel_2D(model_file=stan_model_file, prior_file="stan/graham_prior.stan", figname_base=figname_base)

# set up observations
observations = {"R":[], "proj_density":[]}

for f in HMQ_files:
    SL.logger.info(f"Loading file: {f}")
    hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
    try:
        idx = hmq.get_idx_in_vec(analysis_params["bh_binary"]["target_semimajor_axis"]["value"], hmq.semimajor_axis_of_snapshot)
    except ValueError:
        SL.logger.warning(f"No snapshot data prior to merger! The semimajor_axis_of_snapshot attribute is: {hmq.semimajor_axis_of_snapshot}. This run will not form part of the analysis.")
        continue
    except AssertionError:
        SL.logger.warning(f"Trying to search for value {analysis_params['bh_binary']['target_semimajor_axis']['value']}, but an AssertionError was thrown. The array bounds are {min(hmq.semimajor_axis_of_snapshot)} - {max(hmq.semimajor_axis_of_snapshot)}. This run will not form part of the analysis.")
        continue
    r = cmf.mathematics.get_histogram_bin_centres(hmq.radial_edges)
    observations["R"].append(r)
    observations["proj_density"].append(list(hmq.projected_mass_density.values())[idx])

graham_model.obs = observations


if args.random_sample is not None:
    graham_model.random_obs_select(args.random_sample, "name")

SL.logger.info(f"Number of simulations with usable data: {graham_model.num_groups}")
assert graham_model.num_groups >= analysis_params["stan"]["min_num_samples"]

graham_model.transform_obs("R", "log10_R", lambda x: np.log10(x))
graham_model.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))
graham_model.transform_obs("log10_proj_density", "log10_proj_density_mean", lambda x: np.nanmean(x, axis=0))
graham_model.transform_obs("log10_proj_density", "log10_proj_density_std", lambda x: np.nanstd(x, axis=0))

graham_model.collapse_observations(["log10_R", "log10_proj_density_mean", "log10_proj_density_std"])

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
    fig, ax = plt.subplots(1,1, figsize=full_figsize)
    ax.set_ylim(-1, 15.1)
    ax.set_xlabel("R/kpc")
    ax.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
    ax.set_xscale("log")
    graham_model.prior_plot("R", "log10_proj_density_mean", xmodel="R", ymodel="projected_density", yobs_err="log10_proj_density_std", ax=ax)

    # plot latent parameter prior distributions
    fig, ax = cmf.plotting.create_odd_number_subplots(2,3, fkwargs={"figsize":full_figsize})
    ax[0].set_xscale("log")
    ax[3].set_xscale("log")
    ax[4].set_xscale("log")
    latent_qtys = ["r_b", "Re", "I_b", "g", "n"]
    graham_model.plot_generated_quantity_dist(latent_qtys, xlabels=[r"$r_\mathrm{b}$/kpc", r"$R_\mathrm{e}$/kpc", r"$\Sigma_\mathrm{b}/(10^{9}$M$_\odot$/kpc$^2)$", r"$\gamma$", r"$n$"], ax=ax)
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
    graham_model.sample_model(data=stan_data, sample_kwargs=analysis_params["stan"]["sample_kwargs"])

    graham_model.determine_loo("log10_surf_rho_posterior")

    # parameter corner plots
    var_name_map = dict(
        r_b_a = r"$r_{\mathrm{b}, \alpha}$",
        r_b_b = r"$r_{\mathrm{b}, \beta}$",
        Re_a = r"$R_{\mathrm{e},\alpha}$",
        Re_b = r"$R_{\mathrm{e}, \beta}$",
        I_b_a = r"$\Sigma_{\mathrm{b}, \alpha}$",
        I_b_b = r"$\Sigma_{\mathrm{b}, \beta}$",
        g_a = r"$\gamma_\alpha$",
        g_b = r"$\gamma_\beta$",
        n_a = r"$n_\alpha$",
        n_b = r"$n_\beta$"
    )
    labeller = MapLabeller(var_name_map)

    '''
    graham_model.parameter_plot(["r_b_mean", "r_b_var", "Re_mean", "Re_var"], labeller=labeller)
    graham_model.parameter_plot(["I_b_mean", "I_b_var", "a_mean", "a_var"], labeller=labeller)
    graham_model.parameter_plot(["g_mean", "g_var", "n_mean", "n_var"], labeller=labeller)
    '''
    #graham_model.parameter_plot(["r_b_a", "r_b_b", "Re_a", "Re_b"], labeller=labeller)
    #graham_model.parameter_plot(["I_b_a", "I_b_b", "a_a", "a_b"], labeller=labeller)
    #graham_model.parameter_plot(["g_a", "g_b", "n_a", "n_b"], labeller=labeller)

    # posterior predictive check
    fig, ax = plt.subplots(1,1, figsize=full_figsize)
    ax.set_xlabel(r"log($R$/kpc)")
    ax.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
    graham_model.posterior_plot("log10_R", "log10_proj_density_mean", "log10_surf_rho_posterior", yobs_err="log10_proj_density_std", ax=ax)
    #graham_model.posterior_plot("log10_R", "log10_proj_density", "log10_surf_rho_posterior", ax=ax)

    #graham_model.print_parameter_percentiles(["r_b_mean", "r_b_var", "Re_mean", "Re_var", "I_b_mean", "I_b_var", "g_mean", "g_var", "n_mean", "n_var", "a_mean", "a_var"])

    # plot latent parameter distributions
    #fig, ax = cmf.plotting.create_odd_number_subplots(2,3, fkwargs={"figsize":full_figsize})
    fig, ax = plt.subplots(2,3, figsize=full_figsize)
    ax = np.concatenate(ax).flatten()
    ax[3].set_xscale("log")
    latent_qtys = ["r_b_posterior", "Re_posterior", "I_b_posterior", "g_posterior", "n_posterior", "a_posterior"]
    graham_model.plot_generated_quantity_dist(latent_qtys, xlabels=[r"$r_\mathrm{b}$/kpc", r"$R_\mathrm{e}$/kpc", r"$\Sigma_\mathrm{b}/(10^{9}$M$_\odot$/kpc$^2)$", r"$\gamma$", r"$n$", r"$a$"], ax=ax)
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
    
    graham_model.print_parameter_percentiles(latent_qtys)

#plt.show()


