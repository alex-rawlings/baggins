import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import seaborn as sns
import cm_functions as cmf
import ketjugw

myr = ketjugw.units.yr * 1e6

marker_kw = {"marker":".", "edgecolor":"k", "linewidth":0.3}

def _do_linear_fitting(t, y, t0, tspan, return_idxs=False):
    """
    Wrapper around scipy's curve_fit, with start and end indices of the values 
    to fit from an array also calculated.

    Parameters
    ----------
    t : np.ndarray
        times, a subset of which the linear fit will be performed over
    y : np.ndarray
        corresponding y data, such that y = f(t)
    t0 : float
        time to begin the fit (must be same units as t)
    tspan : float
        duration over which the fit should be performed
    return_idxs : bool, optional
        return the array indices corresponding to t0 and t0+tspan, by default 
        False

    Returns
    -------
    popt : np.ndarray
        optimal parameter values [a,b], such that y = a*t+b
    idxs : list, optional
        indices corresponding to [t0, t0+tspan] if return_idxs is True
    """
    #determine index of t0 in t
    t0idx = np.argmax(t0 < t)
    tfidx = np.argmax(t0+tspan < t)
    #error when t0+tspan==t[-1]
    if t0+tspan >= t[-1]:
        tfidx = -1
        warnings.warn("Analytical fit to binary evolution done to the end of the time data -> proceed with caution!")
    #print(tfidx-t0idx)
    # assume dy/dt is a approx. linear
    '''popt, pcov = scipy.optimize.curve_fit(lambda x, a, b: a*x+b,
                                          t[t0idx:tfidx], y[t0idx:tfidx])'''
    res = scipy.stats.linregress(t[t0idx:tfidx], y[t0idx:tfidx])
    if return_idxs:
        return res, [t0idx, tfidx]
    else:
        return res


data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/"
bhfiles = cmf.utils.get_ketjubhs_in_dir(data_path)

cube_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes/A-C-3.0-0.05/"
cube_files = cmf.utils.get_files_in_dir(cube_path)


fig, ax = plt.subplots(1,1)
ax.set_xlabel("t/Myr")
ax.set_ylabel(r"$HG\rho/\sigma$")

#data = {"t":[], "t_err_L":[], "t_err_U":[], "inv_a":[], "inv_a_err_L":[], "inv_a_err_U":[], "grad":[], "name":[]}
data = {"t":[], "t_err":[], "inv_a":[], "inv_a_err":[], "grad":[], "name":[]}

for j, (bhfile, cubefile) in enumerate(zip(bhfiles, cube_files)):
    print(bhfile)
    cdc = cmf.analysis.ChildSimData.load_from_file(cubefile)
    bh1, bh2, merged = cmf.analysis.get_bound_binary(bhfile)
    orbit_params = ketjugw.orbital_parameters(bh1, bh2)
    # divide the evolution into segments
    time_vals = np.arange(0, 1e4, 10)
    time_duration = 50
    for i, t0 in enumerate(time_vals):
        #print(i)
        if t0+time_duration < orbit_params["t"][-1]/myr and t0>orbit_params["t"][0]/myr:
            res, fit_idxs = _do_linear_fitting(orbit_params["t"]/myr, ketjugw.units.pc/orbit_params["a_R"], t0=t0, tspan=time_duration, return_idxs=True)
            if res.rvalue**2 > 0.95 and orbit_params["a_R"][fit_idxs[1]]/ketjugw.units.pc > 15 and cdc.r_hard_time <= t0:
                idxs = np.r_[fit_idxs[0]:fit_idxs[1]]
                if False:
                    #ax.scatter(orbit_params["t"][idxs]/myr, ketjugw.units.pc/orbit_params["a_R"][idxs])
                    ax.hist(ketjugw.units.pc/orbit_params["a_R"][idxs], bins=20)
                    ax.set_xlabel("pc/a")
                    plt.show()
                    quit()
                '''data["t"].append(np.nanmedian(orbit_params["t"][idxs]/myr))
                data["t_err_L"].append(np.nanquantile(orbit_params["t"][idxs]/myr, 0.25))
                data["t_err_U"].append(np.nanquantile(orbit_params["t"][idxs]/myr, 0.75))
                data["inv_a"].append(np.nanmedian(ketjugw.units.pc / orbit_params["a_R"][idxs]))
                data["inv_a_err_L"].append(np.nanquantile(ketjugw.units.pc / orbit_params["a_R"][idxs], 0.25))
                data["inv_a_err_U"].append(np.nanquantile(ketjugw.units.pc / orbit_params["a_R"][idxs], 0.75))'''
                data["t"].append(np.nanmean(orbit_params["t"][idxs]/myr))
                data["t_err"].append(np.nanstd(orbit_params["t"][idxs]/myr))
                data["inv_a"].append(np.nanmean(ketjugw.units.pc / orbit_params["a_R"][idxs]))
                data["inv_a_err"].append(np.nanstd(ketjugw.units.pc / orbit_params["a_R"][idxs]))
                data["grad"].append(res.slope)
                data["name"].append(j+1)
if True:
    df = pd.DataFrame(data)
    df.to_pickle("pickle/H_G_rho_sigma.pickle")
    print(df)
sns.jointplot(data=df, x="t", y="grad", hue="name")
#ax.legend()
plt.show()
quit()

