import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baggins as bgs
import ketjugw


parser = argparse.ArgumentParser(description="Run stan model for Quinlan evolution.", allow_abbrev=False)
parser.add_argument(type=str, help="file of observed quantities", dest="obs_file")
parser.add_argument("-p", "--prior", help="Plot for prior", action="store_true", dest="prior")
parser.add_argument("-e", "--extract", help="Extract data", action="store_true", dest="extract")
args = parser.parse_args()

# files to read
data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/"
cube_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes/A-C-3.0-0.05/"


# some function definitions 
def extract_data(pickle_file, min_radius=10):
    # helper functions
    def _get_params_in_timespan(r, r0, rf=None):
        def _get_idx_for_r(_r, rvec):
            return np.argmax(_r > rvec)
        idx0 = _get_idx_for_r(r0, r)
        if rf is None:
            idxf = -1
        else:
            idxf = _get_idx_for_r(rf, r)
        return idx0, idxf
    
    # set up data dict
    dd = {"t":[], "E":[], "a":[], "e":[], "m1":[], "m2":[], "name":[]}
    myr = ketjugw.units.yr * 1e6

    # files to read
    bhfiles = bgs.utils.get_ketjubhs_in_dir(data_path)
    cube_files = bgs.utils.get_files_in_dir(cube_path)
    print("Extracting new dataset...")
    for j, (bhfile, cubefile) in enumerate(zip(bhfiles, cube_files), start=1):
        print(bhfile)
        cdc = bgs.analysis.ChildSimData.load_from_file(cubefile)
        bh1, bh2, merged = bgs.analysis.get_bound_binary(bhfile)
        orbit_energy = ketjugw.orbital_energy(bh1, bh2)
        orbit_params = ketjugw.orbital_parameters(bh1, bh2)
        idx_0, idx_f = _get_params_in_timespan(orbit_params["a_R"]/ketjugw.units.pc, cdc.r_hard, min_radius)
        idxs = np.r_[idx_0:idx_f]
        dd["t"].extend(orbit_params["t"][idxs]/myr)
        dd["E"].extend(orbit_energy[idxs])
        dd["a"].extend(orbit_params["a_R"][idxs])
        dd["e"].extend(orbit_params["e_t"][idxs])
        dd["m1"].extend(bh1.m[idxs])
        dd["m2"].extend(bh2.m[idxs])
        dd["name"].extend([j for _ in range(len(orbit_energy[idxs]))])
    df = pd.DataFrame(dd)
    df.to_pickle(pickle_file)
    print(f"New data saved to {pickle_file}.")

# extract data if desired
if args.extract:
    extract_data(args.obs_file)


# set up Stan Model
my_stan = bgs.analysis.StanModel("stan/quinlan_peter.stan", "stan/quinlan_peter_prior.stan", args.obs_file, figname_base="stats/quinlan_peter/qp", random_select_obs={"num":40, "group":"name"})

my_stan.categorical_label = "name"
print(my_stan.obs)

# set up data dictionary with data common to both prior and posterior sampling
data = {"N_child": len(np.unique(my_stan.obs.loc[:, "name"]))}
data.update(
    N_tot = len(my_stan.obs.loc[:, "name"]),
    child_id = my_stan.obs.loc[:, "name"],
    N_per_child = [my_stan.random_obs_select_dict["num"] for _ in range(data["N_child"])],
    t = my_stan.obs.loc[:, "t"],
    mass1 = my_stan.obs.loc[:, "m1"],
    mass2 = my_stan.obs.loc[:, "m2"]
)

if args.prior:
    # append initial conditions for ODE solver
    #initial_idxs = np.full(data["N_child"], 0, int)
    initial_idxs = np.cumsum(data["N_per_child"]) - data["N_per_child"][0]
    data.update(
        E0 = my_stan.obs.loc[initial_idxs, "E"],
        ecc0 = my_stan.obs.loc[initial_idxs, "e"]
    )
    my_stan.sample_prior(data=data)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel("E")
    my_stan.prior_plot("t", "E", "t", "E", ax=ax)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel("e")
    my_stan.prior_plot("t", "e", "t", "ecc", ax=ax)

else:
    # append necessary observations for likelihood
    data.update(
        E = my_stan.obs.loc[:, "E"],
        e = my_stan.obs.loc[:, "e"]
    )

    my_stan.sample_model(data=data)
    my_stan.parameter_plot(var_names=["HGp_s_mu", "HGp_s_tau", "K_mu", "K_tau", "sigma_E", "sigma_e"])
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel("pc/a")
    #my_stan.posterior_plot("t", "E", "inv_a_posterior", ax=ax)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel(r"$\Delta$e")
    #my_stan.posterior_plot("t", "delta_e", "delta_e_posterior", ax=ax)


plt.show()
