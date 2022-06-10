import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cmdstanpy import CmdStanModel
import arviz as av
import ketjugw
import cm_functions as cmf

# user input
extract_data = False
plot_data = False
single_child = False
random_select = 500
figsize = (9.0, 6.75)

data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/"
cube_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes/A-C-3.0-0.05/"
pickle_file = "all_data.pickle"

myr = ketjugw.units.yr * 1e6

rng = np.random.default_rng()

# helper functions
def _get_params_in_timespan(r, r0, rf):
    def _get_idx_for_r(_r, rvec):
        return np.argmax(_r > rvec)
    idx0 = _get_idx_for_r(r0, r)
    idxf = _get_idx_for_r(rf, r)
    return idx0, idxf

# set up data dict
dd = {"t":[], "a":[], "name":[]}

# files to read
bhfiles = cmf.utils.get_ketjubhs_in_dir(data_path)
cube_files = cmf.utils.get_files_in_dir(cube_path)

if extract_data:
    for j, (bhfile, cubefile) in enumerate(zip(bhfiles, cube_files), start=1):
        print(bhfile)
        cdc = cmf.analysis.ChildSimData.load_from_file(cubefile)
        bh1, bh2, merged = cmf.analysis.get_bound_binary(bhfile)
        orbit_params = ketjugw.orbital_parameters(bh1, bh2)
        idx_0, idx_f = _get_params_in_timespan(orbit_params["a_R"]/ketjugw.units.pc, cdc.r_hard, 15)
        idxs = np.r_[idx_0:idx_f]
        dd["t"].extend(orbit_params["t"][idxs]/myr)
        dd["a"].extend(orbit_params["a_R"][idxs]/ketjugw.units.pc)
        dd["name"].extend([j for _ in range(idx_f-idx_0)])
    df = pd.DataFrame(dd)
    df.to_pickle(pickle_file)
    print(df)
else:
    df = pd.read_pickle(pickle_file)

if plot_data:
    fig, ax = plt.subplots(1,1)
    for n in np.unique(df.loc[:, "name"]):
        mask = df.loc[:, "name"] == n
        ax.plot(df.loc[mask, "t"], 1/df.loc[mask, "a"], label=n)
    ax.legend()
    plt.show()

if single_child:
    stan_model = CmdStanModel(stan_file="stan/inv_a_t_single.stan")
    mask = df.loc[:, "name"] != 1
    df.drop(df.loc[mask, "name"].index, axis=0, inplace=True)
else:
    stan_model = CmdStanModel(stan_file="stan/inv_a_t_4.stan")

# prepare stan dictionary
if random_select is None:
    mask = [True for _ in range(df.shape[0])]
    l = df.shape[0]
else:
    i_s = []
    counter = 0
    for n in np.unique(df.loc[:, "name"]):
        _mask = df.loc[:, "name"] == n
        ids = df.loc[_mask, "name"].index
        _i_s = rng.integers(int(ids.min()), int(ids.max()), random_select)
        i_s.extend(_i_s)
        counter += 1
    mask = np.r_[i_s]
    l = random_select * counter

print(df.loc[mask, :])

data = dict(
    N_child = len(np.unique(df.loc[mask, "name"])),
    N_tot = l,
    child_id = df.loc[mask, "name"],
    t = df.loc[mask, "t"],
    a = df.loc[mask, "a"]
)

# run stan model
fit = stan_model.sample(data=data, chains=4, iter_sampling=2000, show_progress=True, show_console=False, adapt_delta=1-1e-1, max_treedepth=12)
summary = fit.summary(sig_figs=4)
print(summary)
print(fit.diagnose())

# plot results
#var_names = ["HGp_s", "c", "sigma"]
var_names = ["HGp_s_mu", "HGp_s_tau", "c_mu", "c_tau", "sigma"]

# plot traces
av.plot_trace(fit, var_names=var_names, figsize=figsize)
fig1 = plt.gcf()
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "stats/stan/stan_chain_all.png"), fig=fig1)
plt.close()


fit_av = av.from_cmdstanpy(fit)
ax1 = av.plot_pair(fit_av, var_names=var_names, kind="scatter", divergences=True, marginals=True, scatter_kwargs={"marker":".", "markeredgecolor":"k", "markeredgewidth":0.5, "alpha":0.2}, figsize=figsize)
ax1 = av.plot_pair(fit_av, var_names=var_names, kind="kde", ax=ax1, point_estimate="mode", marginals=True, kde_kwargs={"contour_kwargs":{"linewidths":0.5}})

fig1 = plt.gcf()
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "stats/stan/stan_pairs_all.png"), fig=fig1)
plt.close()

