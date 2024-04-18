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
    def _get_params_in_timespan(r, r0, rf):
        def _get_idx_for_r(_r, rvec):
            return np.argmax(_r > rvec)
        idx0 = _get_idx_for_r(r0, r)
        idxf = _get_idx_for_r(rf, r)
        return idx0, idxf
    
    # set up data dict
    dd = {"t":[], "a":[], "e":[], "name":[]}
    myr = ketjugw.units.yr * 1e6

    # files to read
    bhfiles = bgs.utils.get_ketjubhs_in_dir(data_path)
    cube_files = bgs.utils.get_files_in_dir(cube_path)
    print("Extracting new dataset...")
    for j, (bhfile, cubefile) in enumerate(zip(bhfiles, cube_files), start=1):
        print(bhfile)
        cdc = bgs.analysis.ChildSimData.load_from_file(cubefile)
        bh1, bh2, merged = bgs.analysis.get_bound_binary(bhfile)
        orbit_params = ketjugw.orbital_parameters(bh1, bh2)
        idx_0, idx_f = _get_params_in_timespan(orbit_params["a_R"]/ketjugw.units.pc, cdc.r_hard, min_radius)
        idxs = np.r_[idx_0:idx_f]
        dd["t"].extend(orbit_params["t"][idxs]/myr)
        dd["a"].extend(orbit_params["a_R"][idxs]/ketjugw.units.pc)
        dd["e"].extend(orbit_params["e_t"][idxs])
        dd["name"].extend([j for _ in range(idx_f-idx_0)])
    df = pd.DataFrame(dd)
    df.to_pickle(pickle_file)
    print(f"New data saved to {pickle_file}.")

# extract data if desired
if args.extract:
    extract_data(args.obs_file)

# set up Stan Model
my_stan = bgs.analysis.StanModel("stan/quinlan_k_2.stan", "stan/quinlan_prior_k_2.stan", args.obs_file, figname_base="stats/quinlan_stan/quinlan", random_select_obs={"num":20, "group":"name"})

# do data transformations
my_stan.transform_obs("a", "inv_a", lambda a:1/a)

def _delta_e_maker():
    e0 = []
    delta_e = []
    offset = 0
    for n in np.unique(my_stan.obs.loc[:, "name"]):
        mask = my_stan.obs.loc[:, "name"] == n
        e0 = my_stan.obs.loc[mask, "e"]
        delta_e.extend(my_stan.obs.loc[mask, "e"]-e0[offset])
        offset += my_stan.random_obs_select_dict["num"]
    return delta_e

my_stan.obs["delta_e"] = _delta_e_maker()

my_stan.categorical_label = "name"
print(my_stan.obs)

"""fig, ax = plt.subplots(1,1)
for n in np.unique(my_stan.obs.loc[:, "name"]):
    mask = n == my_stan.obs.loc[:, "name"]
    ax.scatter(my_stan.obs.loc[mask, "t"], my_stan.obs.loc[mask, "inv_a"], marker=".")
ax.set_xlabel("t/Myr")
ax.set_ylabel(r"1/a | (a$_h$ < a/pc < 10)")
plt.savefig(f"{bgs.FIGDIR}/{my_stan.figname_base}_raw.png")
plt.show()
quit()"""

if args.prior:
    # synthetic data
    data = {"N_tot":500, "N_child":10}
    data.update(
        dict(
                child_id = [
                    i for i in np.arange(1, data["N_child"]+1) for j in range(int(data["N_tot"]/data["N_child"]))
                ],
                t = np.linspace(0.9*np.min(my_stan.obs.loc[:,"t"]), 1.1*np.max(my_stan.obs.loc[:,"t"]), data["N_tot"])
        )
    )
    my_stan.sample_prior(data=data)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel("pc/a")
    my_stan.prior_plot("t", "inv_a", "t", "inv_a_prior", ax=ax)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel(r"$\Delta$e")
    my_stan.prior_plot("t", "delta_e", "t", "del_e_prior", ax=ax)

else:
    data = dict(
        N_child = len(np.unique(my_stan.obs.loc[:, "name"])),
        N_tot = len(my_stan.obs.loc[:, "name"]),
        child_id = my_stan.obs.loc[:, "name"],
        t = my_stan.obs.loc[:, "t"],
        a = my_stan.obs.loc[:, "a"],
        delta_e = my_stan.obs.loc[:, "delta_e"]
    )
    my_stan.sample_model(data=data)
    my_stan.parameter_plot(var_names=["HGp_s_mu", "HGp_s_tau", "K_mu", "K_tau", "sigma_inv_a", "sigma_e"])
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel("pc/a")
    my_stan.posterior_plot("t", "inv_a", "inv_a_posterior", ax=ax)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("t/Myr")
    ax.set_ylabel(r"$\Delta$e")
    my_stan.posterior_plot("t", "delta_e", "delta_e_posterior", ax=ax)


plt.show()
