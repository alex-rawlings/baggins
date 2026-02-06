import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import ketjugw
import baggins as bgs
from cmdstanpy import CmdStanModel

# ---------------------------------------------------
# 1. Compile the Stan model
# ---------------------------------------------------
stan_file = "/users/arawling/projects/collisionless-merger-sample/baggins/stan/gaussian-process/gp_analytic_deriv.stan"
model = CmdStanModel(stan_file=stan_file)

# ---------------------------------------------------
# 2. Example synthetic data
# ---------------------------------------------------
kfile = bgs.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/ecc-chaos/e-090/8M/D_8M_a-D_8M_b-3.720-0.279/output")[0]
bh1, bh2, _ = bgs.analysis.get_bound_binary(kfile)
orbit_pars = ketjugw.orbital_parameters(bh1, bh2)

# times in Myr
ta = orbit_pars["t"][::1000] / bgs.general.units.Myr
Na = len(ta)
Np = 10 * Na
print(f"There are {len(ta)} training points")
tp = np.linspace(np.min(ta), np.max(ta), Np)

# fake data: shrinking a, decaying e
a_obs = np.log10(orbit_pars["a_R"][::1000] / 1e3/ bgs.general.units.kpc)  # kpc
e_obs = orbit_pars["e_t"][::1000]

data = {
    "N1": Na,
    "x1": ta,
    "y1": a_obs,
    "N2": Np,
    "x2": tp,
    "m1_msun": 1e8,
    "m2_msun": 5e7,
    "seconds_per_time_unit": 1.0e6 * 365.25 * 86400,  # time units = Myr
}

# ---------------------------------------------------
# 3. Fit the model
# ---------------------------------------------------
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_sampling=1000,
    iter_warmup=500,
    seed=123,
)

# ---------------------------------------------------
# 4. Extract posterior draws
# ---------------------------------------------------
a_pred = fit.stan_variable("powy")       # (draws, Np)
#e_pred = fit.stan_variable("e_pred")       # (draws, Np)
s_total = fit.stan_variable("powdy")     # (draws, Np)
#s_peters = fit.stan_variable("s_peters")   # (draws, Np)
#s_other = fit.stan_variable("s_other")     # (draws, Np)

# ---------------------------------------------------
# 5. Plot semimajor axis and eccentricity fits
# ---------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].set_yscale("symlog", linthresh=1e-9)

# TODO need this still
dadt_gw, dedt_gw = ketjugw.peters_derivatives(a_pred*bgs.general.units.kpc, 0.999, bh1.m[0], bh2.m[0])
dadt_gw /= (bgs.general.units.kpc / bgs.general.units.Myr)

# Semimajor axis
median_a = np.median(a_pred, axis=0)
axes[0].plot(tp, median_a[Na:], color="C0", label="GP median")
az.plot_hdi(tp, a_pred[:, Na:], hdi_prob=0.5, color="C0", ax=axes[0], hdi_kwargs={"skipna":True})
axes[0].scatter(ta, 10**a_obs, color="k", s=20, label="data")
axes[0].set_ylabel("a [kpc]")
axes[0].legend()

'''# Eccentricity
median_e = np.median(e_pred, axis=0)
axes[1].plot(tp, median_e, color="C1", label="GP median")
az.plot_hdi(tp, e_pred, hdi_prob=0.5, color="C1", ax=axes[1])
axes[1].scatter(te, e_obs, color="k", s=20, label="data")
axes[1].set_xlabel("Time [Myr]")
axes[1].set_ylabel("e")
axes[1].legend()'''

plt.tight_layout()
plt.savefig("gp_dadt1.png", dpi=300)

# ---------------------------------------------------
# 6. Plot hardening rates
# ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

# Plot median lines
#ax.plot(tp, np.median(s_total[Na:], axis=0), color="C0", label=r"$s_{\rm total}$")
#ax.plot(tp, np.median(s_peters, axis=0), color="C1", label=r"$s_{\rm Peters}$")
#ax.plot(tp, np.median(s_other, axis=0), color="C2", label=r"$s_{\rm other}$")

# HDI bands
#az.plot_hdi(tp, s_total, hdi_prob=0.5, color="C0", ax=ax)
#az.plot_hdi(tp, dadt_gw[:, Na:], hdi_prob=0.5, color="C1", ax=ax)
#az.plot_hdi(tp, s_total-dadt_gw[:, Na:], hdi_prob=0.5, color="C2", ax=ax)
mask = ~np.isnan(s_total)
mask = np.logical_and(
        dadt_gw[:, Na:] > np.nanquantile(dadt_gw[:, Na:], 1e-2),
        dadt_gw[:, Na:] < np.nanquantile(dadt_gw[:, Na:], 1-1e-2)
)
#az.plot_kde(dadt_gw[:, Na:][mask], s_total[mask]-dadt_gw[:, Na:][mask], ax=ax)
az.plot_kde(dadt_gw[:, Na:][mask], ax=ax, plot_kwargs={"c":"C1", "label":"GW"})
az.plot_kde(s_total-dadt_gw[:, Na:], ax=ax, plot_kwargs={"c":"C2", "label":"Other"})
az.plot_kde(s_total, ax=ax, plot_kwargs={"c":"C3", "label":"Total"})
ax.set_xscale("symlog")

ax.set_xlabel("dadt_gw")
ax.set_ylabel(r"dadt_other")
ax.legend()

plt.tight_layout()
plt.savefig("gp_dadt2.png", dpi=300)
