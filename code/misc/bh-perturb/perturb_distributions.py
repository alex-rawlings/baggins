import os.path
import numpy as np
import pandas as pd
import scipy.stats 
import matplotlib.pyplot as plt
import seaborn as sns
import cm_functions as cmf


main_data_path = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/pickle/bh_perturb/"
data_files = [
    "NGCa0524_bhperturb.pickle", "NGCa2986_bhperturb.pickle", "NGCa3348_bhperturb.pickle", "NGCa3607_bhperturb.pickle", "NGCa4291_bhperturb.pickle"
]

#mask the data
time_lower = 4

#set up dataframe for seaborn use
#whilst iterating, also test goodness of fit to normal distribution
fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].set_title("Position")
ax[1].set_title("Velocity")
cols = cmf.plotting.mplColours()
name_list = []
Astat_pos = np.full_like(data_files, np.nan, dtype=float)
Astat_vel = np.full_like(Astat_pos, np.nan)
for i, datafile in enumerate(data_files):
    data_dict = cmf.utils.load_data(os.path.join(main_data_path, datafile))
    time_mask = data_dict["times"] > time_lower
    displacement = cmf.mathematics.radial_separation(data_dict["diff_x"])[time_mask]
    vel_mag = cmf.mathematics.radial_separation(data_dict["diff_v"])[time_mask]
    names = [data_dict["galaxy_name"]] * len(displacement)
    tempdict = {"radial":displacement, "velmag":vel_mag, "name":names}
    #perform the Anderson-Darling test for normality
    print("Anderson Test for {}".format(data_dict["galaxy_name"]))
    name_list.append(data_dict["galaxy_name"])
    for Astatarr, vals in zip((Astat_pos, Astat_vel), (displacement, vel_mag)):
        Astatarr[i], critval, siglev = scipy.stats.anderson(vals, dist="norm")
    #print("----------------------")
    if i==0:
        dataframe = pd.DataFrame.from_dict(tempdict)
    else:
        dataframe2 = pd.DataFrame.from_dict(tempdict)
        dataframe = dataframe.append(dataframe2, ignore_index=True)
for i, c in enumerate(critval):
    ax[0].axhline(c, c=cols[i], label="{}%".format(siglev[i]))
    ax[1].axhline(c, c=cols[i])
for i, arr in enumerate((Astat_pos, Astat_vel)):
    ax[i].scatter(name_list, arr, c="k", label=r"$A^{*2}$", zorder=10)
    plt.sca(ax[i])
    plt.xticks(rotation=45)
ax[0].legend()

#plot the displacement and velocity magnitudes, with a kernel-density estimate
p = sns.jointplot(data=dataframe, x="radial", y="velmag", hue="name")
p.set_axis_labels("Radial Displacement [kpc]", "Velocity Magnitude [km/s]")
fig.savefig(os.path.join(cmf.FIGDIR, "brownian/anderson.png"))
p.figure.savefig(os.path.join(cmf.FIGDIR, "brownian/all_brownian.png"))
plt.show()