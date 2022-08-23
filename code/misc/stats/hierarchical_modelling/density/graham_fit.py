import argparse
import os.path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cm_functions as cmf
import ketjugw
import pygad


parser = argparse.ArgumentParser(description="Run stan model for Quinlan evolution.", allow_abbrev=False)
parser.add_argument(type=str, help="file of observed quantities", dest="file")
parser.add_argument("-p", "--prior", help="Plot for prior", action="store_true", dest="prior")
parser.add_argument("-l", "--load", type=str, help="Load previous stan file", dest="load_file", default=None)
parser.add_argument("-e", "--extract", help="Extract data", action="store_true", dest="extract")
args = parser.parse_args()


myr = ketjugw.units.yr * 1e6
# files to read
data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/"
cube_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes/A-C-3.0-0.05/"
figname_base="stats/graham_density/graham_density"
#radial_bins = np.geomspace(0.2, 20, 51)
radial_bins = np.arange(0, 50, 0.19)
radial_bin_centres = cmf.mathematics.get_histogram_bin_centres(radial_bins)


def extract_data(pickle_file, a_val=10):
    dd = {"R":[], "Sigma":None, "name":[]}
    # files to read
    bhfiles = cmf.utils.get_ketjubhs_in_dir(data_path)
    cube_files = cmf.utils.get_files_in_dir(cube_path)
    print("Extracting new dataset...")
    for j, bhfile in enumerate(bhfiles):
        #if j ==0: continue
        print(bhfile)
        bh1, bh2, merged = cmf.analysis.get_bound_binary(bhfile)
        orbit_params = ketjugw.orbital_parameters(bh1, bh2)
        idx = np.argmax(a_val > orbit_params["a_R"]/ketjugw.units.pc)
        if idx == 0:
            warnings.warn(f"Returning first index in orbital parameters! 'a' value of {a_val}pc may not be in the array!")
            print(f"max a: {np.max(orbit_params['a_R']/ketjugw.units.pc)}")
            print(f"min a: {np.min(orbit_params['a_R']/ketjugw.units.pc)}")
        t = orbit_params["t"][idx]/myr
        print(f"Time to find: {t:.3f} Myr (--> {orbit_params['a_R'][idx]/ketjugw.units.pc:.3f}pc)")
        snaplist = cmf.utils.get_snapshots_in_dir(os.path.join(data_path, f"{j:03d}/output"))
        snap_idx = cmf.analysis.snap_num_for_time(snaplist, t)
        snap = pygad.Snapshot(snaplist[snap_idx], physical=True)
        eff_rad, vsigRe, vsigr, _Sigma = cmf.analysis.projected_quantities(snap, obs=3, r_edges=radial_bins)
        Sigma = list(_Sigma.values())[0]
        if j==0:
            dd["Sigma"] = Sigma
        else:
            dd["Sigma"] = np.hstack((dd["Sigma"], Sigma))
        dd["R"].extend(radial_bin_centres)
        print(dd["Sigma"].shape)
        dd["name"].extend([j+1 for _ in range(Sigma.shape[-1])])
        pygad.gc_full_collect()
        snap.delete_blocks()
        del snap
    cmf.utils.save_data(dd, pickle_file)
    print(f"New data saved to {pickle_file}.")

# extract data if desired
if args.extract:
    extract_data(args.file)
else:
    obs = cmf.utils.load_data(args.file)


if args.load_file is not None:
    my_stan = cmf.analysis.StanModel.load_fit(args.load_file, obs_file=args.file, figname_base=figname_base)
else:
    my_stan = cmf.analysis.StanModel(model_file="stan/graham.stan", prior_file="stan/graham_prior.stan", obs_file=args.file, figname_base=figname_base)


my_stan.categorical_label = "name"
my_stan.transform_obs("Sigma", "log10_Sigma", lambda x: np.log10(x))
my_stan.transform_obs("Sigma", "log10_Sigma_mean", lambda x: np.nanmean(np.log10(x), axis=0))
my_stan.transform_obs("Sigma", "log10_Sigma_std", lambda x: np.nanstd(np.log10(x), axis=0))


data = {"a": 10.0}


'''obs_file = "/scratch/pjohanss/arawling/testing/alex_density.pickle"
observations = cmf.utils.load_data(obs_file)
my_stan.obs = pd.DataFrame(data={"r":observations["run2"]["x"], "surf_rho":np.log10(np.median(observations["run2"]["density"]["0.0"], axis=0)), "group":["AD" for _ in range(len(observations["run2"]["x"]))]})
my_stan.categorical_label = "group"'''


if args.prior:
    N = 1000
    data.update(dict(
                N_tot = N,
                R = np.geomspace(1e-2, 1e2, N)
    ))
    my_stan.sample_prior(data, sample_kwargs={"adapt_delta":1-1e-3})

    fig, ax = plt.subplots(1,1)
    ax.set_ylim(-1, 15.1)
    ax.set_xlabel("r/kpc")
    ax.set_ylabel(r"log($\Sigma(r)$/(M$_\odot$/kpc$^2$))")
    ax.set_xscale("log")
    my_stan.prior_plot("R", "log10_Sigma_mean", xmodel="R", ymodel="projected_density", ax=ax)
else:
    data.update(dict(
                N_tot = my_stan.obs_len,
                R = my_stan.obs["R"],
                N_child = len(np.unique(my_stan.obs["name"])),
                child_id = my_stan.obs["name"],
                log10_surf_rho = my_stan.obs["log10_Sigma_mean"],
                log10_surf_rho_err = my_stan.obs["log10_Sigma_std"]
    ))
    
    my_stan.sample_model(data=data)

    if True:
        fig, ax = plt.subplots(1,1)
        ax.set_xscale("log")
        ax.set_xlabel("R/kpc")
        ax.set_ylabel(r"$\Sigma$/(M$_\odot$kpc$^{-2}$)")
        my_stan.posterior_plot("R", "log10_Sigma_mean", "log10_surf_rho_posterior", yobs_err="log10_Sigma_std", ax=ax)
    if False:
        my_stan.parameter_plot(["r_b_a", "r_b_b", "Re_a", "Re_b"])
        my_stan.parameter_plot(["I_b_a", "I_b_b", "g_a", "g_b", "n_a", "n_b"])
    if True:
        my_stan.print_parameter_percentiles(["r_b_a", "r_b_b", "Re_a", "Re_b", "I_b_a", "I_b_b", "g_a", "g_b", "n_a", "n_b"])
    
    plt.show()


plt.show()