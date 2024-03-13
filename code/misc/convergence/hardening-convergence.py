import os.path
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


paramfile_base = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/resolution-convergence/hardening/"

def rho_sigma(snaplist, t, extent=None):
    """
    A hacked version of bgs.analysis.get_G_rho_per_sigma that returns rho and 
    sigma separately
    """
    ts = np.full((2), np.nan)
    inner_density = np.full_like(ts, np.nan)
    inner_sigma = np.full_like(ts, np.nan)
    idx = bgs.analysis.snap_num_for_time(snaplist, t, method="floor", units="Myr")
    star_count = 0
    for i in range(2):
        snap = pygad.Snapshot(snaplist[idx+i], physical=True)
        extent_mask = pygad.BallMask(extent, center=pygad.analysis.center_of_mass(snap.bh))
        star_count += len(snap.stars[extent_mask]["mass"])
        ts[i] = bgs.general.convert_gadget_time(snap, new_unit="Myr")
        rho_temp, sigma_temp = bgs.analysis.get_inner_rho_and_sigma(snap, extent)
        rho_units = rho_temp.units
        sigma_units = sigma_temp.units
        inner_density[i], inner_sigma[i] = rho_temp, sigma_temp
    grad_rho = np.diff(inner_density)/np.diff(ts)
    grad_sig = np.diff(inner_sigma)/np.diff(ts)
    f_rho = scipy.interpolate.interp1d(ts, inner_density)
    f_sigma = scipy.interpolate.interp1d(ts, inner_sigma)
    return pygad.UnitScalar(f_rho(t), rho_units), pygad.UnitScalar(f_sigma(t), sigma_units), grad_rho, grad_sig, star_count/2

#set up the figures
fig, ax = plt.subplots(4,3,sharex="all", squeeze=False)
for axi in ax[-1,:]:
    axi.set_xlabel(r"$N_\star$")
ax[0,0].set_ylabel(r"d$(1/a)$/d$t$ [1/(pc yr)]")
ax[0,1].set_ylabel(r"$H$")
ax[0,2].set_ylabel(r"$G\rho/\sigma$ [1/(pc yr)]")
ax[1,0].set_ylabel(r"$\rho$ [M$_\odot$/pc$^3$]")
ax[1,1].set_ylabel(r"$\sigma$ [km/s]")
ax[1,2].set_ylabel(r"$K$")
ax[2,0].set_ylabel(r"$\Delta\rho/\Delta t$ [M$_\odot$pc$^3$/Myr]")
ax[2,1].set_ylabel(r"$\Delta\sigma/\Delta t$ [km/s/Myr]")
ax[2,2].set_ylabel(r"$N_\star(<r_\mathrm{infl})$")
ax[3,0].set_ylabel(r"$r_\mathrm{infl}$ [pc]")

if False:
    data = dict()

    for pfR, n in zip(("DE-030-0005r1.py", "DE-030-0005r2.py", "DE-030-0005r5.py"), ("1", "2", "5")):
        pf = os.path.join(paramfile_base, pfR)
        print("Main parameter file: {}".format(pf))
        data[n] = dict()
        data[n]["nstar"]  = np.full(10, np.nan)
        data[n]["H_arr"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["K_arr"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["da_arr"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["Gps_arr"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["rho_arr"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["sig_arr"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["rho_grad"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["sig_grad"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["star_count"] = np.full_like(data[n]["nstar"], np.nan)
        data[n]["rinfl"] = np.full_like(data[n]["nstar"], np.nan)
        for i in range(10):
            print("  Child {}".format(i))
            perturb_idx = "{:03d}".format(i)
            bhb = bgs.analysis.BHBinary(pf, perturbID=perturb_idx, gr_safe_radius=10)
            data[n]["nstar"] = bhb.starcount
            data[n]["H_arr"][i] = bhb.H
            data[n]["K_arr"][i] = bhb.K
            data[n]["Gps_arr"][i] = bhb.G_rho_per_sigma
            data[n]["rinfl"][i] = bhb.r_infl
            data[n]["da_arr"][i] = bhb.H * bhb.G_rho_per_sigma
            data[n]["rho_arr"][i], data[n]["sig_arr"][i], data[n]["rho_grad"][i], data[n]["sig_grad"][i], data[n]["star_count"][i] = rho_sigma(bhb.snaplist, bhb.r_hard_time, bhb.r_infl)
    bgs.utils.save_data(data, "hardening-convergence.pickle")
else:
    data = bgs.utils.load_data("hardening-convergence.pickle")

#plot
for r in data.keys():
    for k,v in data[r].items():
        if k =="nstar":
            nstar_med = np.nanmedian(v)
            continue
        median_val = np.nanmedian(v)
        yerr = np.full((2,1), np.nan, dtype=float)
        yerr[0,0], yerr[1,0] = median_val - np.nanquantile(v, 0.25, axis=-1), np.nanquantile(v, 0.75, axis=-1) - median_val
        if k=="da_arr": axi=ax[0,0]
        elif k=="H_arr": axi=ax[0,1]
        elif k=="Gps_arr": axi=ax[0,2]
        elif k=="rho_arr": axi=ax[1,0]
        elif k=="sig_arr": axi=ax[1,1]
        elif k=="K_arr": axi=ax[1,2]
        elif k=="rho_grad": axi=ax[2,0]
        elif k=="sig_grad": axi=ax[2,1]
        elif k=="star_count": axi=ax[2,2]
        else: axi=ax[3,0]
        axi.errorbar(nstar_med, median_val, yerr=yerr, fmt="o")
plt.show()