import os.path
import numpy as np
import scipy.stats, scipy.spatial.transform
import matplotlib.pyplot as plt
import seaborn as sns
import cm_functions as cmf


def plothist_and_best(mu, sigma, ax1, ax2, ax3, nbins=10, label=None, endpoint=1e-3):
    ax1.hist(mu, nbins, histtype="step", density=True)
    h,bins,_ = ax2.hist(sigma, nbins, histtype="step", density=True, label=label)
    best_mean = np.nanmean(mu)
    bincentres = cmf.mathematics.get_histogram_bin_centres(bins)
    best_sd = bincentres[np.argmax(h)]
    distr = scipy.stats.norm(best_mean, best_sd)
    x = np.linspace(distr.ppf(endpoint), distr.ppf(1-endpoint), 100)
    ax3.plot(x, distr.pdf(x), label=label)



main_data_path = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/pickle/bh_perturb/"
data_files = [
    "NGCa0524_bhperturb.pickle", "NGCa2986_bhperturb.pickle", "NGCa3348_bhperturb.pickle", "NGCa3607_bhperturb.pickle", "NGCa4291_bhperturb.pickle"
]


end_points = 0.01
reps = 5000
nbins = 50
time_lower = 4

fig, ax = plt.subplots(3,4, sharex="col", sharey="col", figsize=(8,7))
ax[0,0].set_title("Position Mean")
ax[0,1].set_title("Position SD")
ax[0,2].set_title("Velocity Mean")
ax[0,3].set_title("Velocity SD")

fig2, ax2 = plt.subplots(3,2, sharex="col", sharey="col")
ax2[0,0].set_title("Position")
ax2[0,1].set_title("Velocity")

for i, val in enumerate(("x", "y", "z")):
    ax[i,0].set_xlabel(val)
    ax[i,1].set_xlabel(val)
    ax[i,2].set_xlabel(r"v$_{}$".format(val))
    ax[i,3].set_xlabel(r"v$_{}$".format(val))
    ax2[i,0].set_xlabel(val)
    ax2[i,1].set_xlabel(r"v$_{}$".format(val))

for l, datafile in enumerate(data_files):
    data_dict = cmf.utils.load_data(os.path.join(main_data_path, datafile))
    time_mask = data_dict["times"] > time_lower
    means = dict(
        pos = np.full((reps, 3), np.nan),
        vel = np.full((reps,3), np.nan)
    )
    sds = dict(
        pos = np.full((reps, 3), np.nan),
        vel = np.full((reps,3), np.nan)
    )
    for j in range(reps):
        print("Rotating {:.1f}%              ".format(j/(reps-1)*100), end="\r")
        rotation = scipy.spatial.transform.Rotation.random()
        pos = rotation.apply(data_dict["diff_x"][time_mask])
        vel = rotation.apply(data_dict["diff_v"][time_mask])
        for k in range(3):
            means["pos"][j,k], sds["pos"][j,k] = scipy.stats.norm.fit(pos[:,k])
            means["vel"][j,k], sds["vel"][j,k] = scipy.stats.norm.fit(vel[:,k])
    print("Rotation complete                                      ")

    for i in range(3):
        plothist_and_best(means["pos"][:,i], sds["pos"][:,i], ax[i,0], ax[i,1], ax2[i,0], label=data_dict["galaxy_name"], endpoint=end_points)
        plothist_and_best(means["vel"][:,i], sds["vel"][:,i], ax[i,2], ax[i,3], ax2[i,1], endpoint=end_points)
ax[0,1].legend()
ax2[0,0].legend()
#fig.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/random_rots.png"))
#plt.show()

"""
fig, ax = plt.subplots(3,2,sharex="col")
for i, datafile in enumerate(data_files):
    data_dict = cmf.utils.load_data(os.path.join(main_data_path, datafile))
    for j, key in enumerate(data_dict.keys()):
        if key == "galaxy_name": continue
        for k in range(3):
            #do the kde plot
            p = sns.kdeplot(data_dict[key][:,k], gridsize=100, ax=ax[k,j])
            #fit a Guassian
            params = scipy.stats.norm.fit(data_dict[key][:,k])
            fit_distr = scipy.stats.norm(*params)
            x = np.linspace(fit_distr.ppf(end_points), fit_distr.ppf(1-end_points), 100)
            ax[k,j].plot(x, fit_distr.pdf(x), c=p.lines[-1].get_color())
plt.show()
"""
plt.show()