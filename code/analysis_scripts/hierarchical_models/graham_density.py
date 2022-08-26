import argparse
import os.path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cm_functions as cmf



parser = argparse.ArgumentParser(description="Run stan model for Core-Sersic model.", allow_abbrev=False)
parser.add_argument(type=str, help="Directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-p", "--prior", help="Plot for prior", action="store_true", dest="prior")
parser.add_argument("-l", "--load", type=str, help="Load previous stan file", dest="load_file", default=None)
args = parser.parse_args()

HMQ_files = cmf.utils.get_files_in_dir(args.dir)
with h5py.File(HMQ_files[0], mode="r") as f:
    merger_id = f["/meta"].attrs["merger_id"]
figname_base = f"hierarchical_models/density/graham_density-{merger_id}"

analysis_params = cmf.utils.read_parameters(args.apf)

if args.load_file is not None:
    # load a previous sample for improved performance: no need to resample the
    # likelihood function
    graham_model = cmf.analysis.StanModel.load_fit(args.load_file, figname_base=figname_base)
else:
    # sample
    graham_model = cmf.analysis.StanModel(model_file="stan/graham.stan", prior_file="stan/graham_prior.stan", figname_base=figname_base)


# set up observations
observations = {"R":[], "proj_density":[], "name":[]}
for i, f in enumerate(HMQ_files):
    print(f"Loading file: {f}")
    hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
    r = cmf.mathematics.get_histogram_bin_centres(hmq.radial_edges)
    observations["R"].extend(r)
    idx = hmq.get_idx_in_vec(analysis_params.target_semimajor_axis, hmq.semimajor_axis_of_snapshot)
    if i == 0:
        observations["proj_density"] = list(hmq.projected_mass_density.values())[idx]
    else:
        observations["proj_density"] = np.hstack((observations["proj_density"], list(hmq.projected_mass_density.values())[idx]))
    observations["name"].extend([i+1 for _ in range(len(r))])
graham_model.obs = observations


graham_model.categorical_label = "name"
graham_model.transform_obs("proj_density", "log10_proj_density", lambda x: np.log10(x))
graham_model.transform_obs("proj_density", "log10_proj_density_mean", lambda x: np.nanmean(np.log10(x), axis=0))
graham_model.transform_obs("proj_density", "log10_proj_density_std", lambda x: np.nanstd(np.log10(x), axis=0))


stan_data = {"a": 10.0}

if args.prior:
    # create the push-forward distribution for the prior model
    R_unique = np.unique(graham_model.obs["R"])
    stan_data.update(dict(
        R = R_unique,
        N_tot = len(R_unique)
    ))

    graham_model.sample_prior(data=stan_data, sample_kwargs=analysis_params.stan_sample_kwargs)

    # prior predictive check
    fig, ax = plt.subplots(1,1)
    ax.set_ylim(-1, 15.1)
    ax.set_xlabel("r/kpc")
    ax.set_ylabel(r"log($\Sigma(r)$/(M$_\odot$/kpc$^2$))")
    ax.set_xscale("log")
    graham_model.prior_plot("R", "log10_proj_density_mean", xmodel="R", ymodel="projected_density", yobs_err="log10_proj_density_std", ax=ax)
else:
    # create the push-forward distribution for the posterior model
    stan_data.update(dict(
                N_tot = graham_model.obs_len,
                R = graham_model.obs["R"],
                N_child = len(np.unique(graham_model.obs["name"])),
                child_id = graham_model.obs["name"],
                log10_surf_rho = graham_model.obs["log10_proj_density_mean"],
                log10_surf_rho_err = graham_model.obs["log10_proj_density_std"]
    ))

    analysis_params.stan_sample_kwargs["output_dir"] = os.path.join(cmf.DATADIR, f"stan_files/{merger_id}")
    graham_model.sample_model(data=stan_data, sample_kwargs=analysis_params.stan_sample_kwargs)

    # parameter corner plots
    graham_model.parameter_plot(["r_b_a", "r_b_b", "Re_a", "Re_b"])
    graham_model.parameter_plot(["I_b_a", "I_b_b", "g_a", "g_b", "n_a", "n_b"])

    # posterior predictive check
    fig, ax = plt.subplots(1,1)
    ax.set_xscale("log")
    ax.set_xlabel("R/kpc")
    ax.set_ylabel(r"$\Sigma$/(M$_\odot$kpc$^{-2}$)")
    graham_model.posterior_plot("R", "log10_proj_density_mean", "log10_surf_rho_posterior", yobs_err="log10_proj_density_std", ax=ax)

    graham_model.print_parameter_percentiles(["r_b_a", "r_b_b", "Re_a", "Re_b", "I_b_a", "I_b_b", "g_a", "g_b", "n_a", "n_b"])

plt.show()


