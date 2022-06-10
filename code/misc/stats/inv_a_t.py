import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
from cmdstanpy import CmdStanModel
import arviz as av
import cm_functions as cmf


# read in inverse semimajor axis and time data, and clean
_data = pd.read_pickle("../merger-test/pickle/H_G_rho_sigma.pickle")
_data.dropna(axis=0, inplace=True)

print(_data)
mask = _data.loc[:, "name"] == 1
target = np.mean(_data.loc[mask, 'grad'])
#print(f"Target mean HGp_s: {np.mean(_data.loc[mask, 'grad'])}")
#quit()
# plot the data
fig, ax = plt.subplots(1,1)
ax.set_xlabel("t/Myr")
ax.set_ylabel("pc/a")
#ax.errorbar(_data["t"], _data["inv_a"], xerr=[_data["t"]-_data["t_err_L"], _data["t_err_U"]-_data["t"]], yerr=[_data["inv_a"]-_data["inv_a_err_L"], _data["inv_a_err_U"]-_data["inv_a"]], fmt=".")
ax.errorbar(_data.loc[mask, "t"], _data.loc[mask, "inv_a"], xerr=_data.loc[mask, "t_err"], yerr=_data.loc[mask,"inv_a_err"], fmt=".")
#plt.show()
#quit()

#ax.scatter(_data.loc[mask, "inv_a"][:-1], _data.loc[mask, "inv_a"][1:])
res = scipy.optimize.curve_fit(lambda x,a,b: a*x+b, _data.loc[mask, "inv_a"][:-1], _data.loc[mask, "inv_a"][1:])
print("Scipy result")
print(res)
#plt.show()


# set up stan model
#stan_model = CmdStanModel(stan_file="inv_a_t.stan")
stan_model = CmdStanModel(stan_file="stan/inv_a_t_3.stan")

# prepare stan dictionary
"""
data = dict(
        N_c = 1,
        N_tot = _data.shape[0],
        child_id = _data["name"].to_numpy(),
        t = _data["t"].to_numpy(),
        inv_a = _data["inv_a"].to_numpy(),
        t_sigma = _data["t_err"].to_numpy(),
        inv_a_sigma = _data["inv_a_err"].to_numpy() # really needs to be std, not quantile!
)
"""
"""
data = dict(
        N_c = 1,
        N_tot = np.sum(mask),
        child_id = _data.loc[mask, "name"],
        t = _data.loc[mask, "t"],
        inv_a = _data.loc[mask, "inv_a"],
        t_sigma = _data.loc[mask, "t_err"],
        inv_a_sigma = _data.loc[mask, "inv_a_err"],
)
"""

"""
data = dict(
        N_tot = np.sum(mask),
        t = _data.loc[mask, "t"],
        inv_a = _data.loc[mask, "inv_a"],
        inv_a_err = _data.loc[mask, "inv_a_err"],
)
"""

data = dict(
        N_child = 10,
        N_tot = _data.shape[0],
        child_id = _data.loc[:, "name"],
        t = _data.loc[:, "t"],
        inv_a = _data.loc[:, "inv_a"],
        inv_a_err = _data.loc[:, "inv_a_err"],
)

# run the stan model
fit = stan_model.sample(data=data, chains=4, iter_sampling=10000, show_progress=False, show_console=False)#, max_treedepth=12, adapt_delta=1-1e-5)
summary = fit.summary(sig_figs=4)
print(summary)
print(fit.diagnose())

# plot results
#var_names = ["HGp_s", "intercept"]
var_names = ["HGp_s_mu", "HGp_s_tau", "c_mu", "c_tau"]

# plot traces
av.plot_trace(fit, var_names=var_names)

"""
fit_av = av.from_cmdstanpy(fit, 
                           coords={"N_c":np.arange(data["N_c"]), 
                                   "N_tot":np.arange(data["N_tot"])},
                           dims={"HGp_s":["N_c"], "intercept":["N_c"]})
"""
fit_av = av.from_cmdstanpy(fit)
#ax = av.plot_pair(fit, var_names=var_names, kind="scatter", scatter_kwargs={"marker":".", "markeredgecolor":"k", "markeredgewidth":0.5, "alpha":0.2})
ax1 = av.plot_pair(fit_av, var_names=var_names, kind="scatter", point_estimate="mean", divergences=True, marginals=True)#, coords={"N_c":np.array([0,1]), "N_tot":np.array([0])})
print(ax1.shape)
ax1[1,0].axvline(target, c="tab:red")

plt.show()