import os.path
import numpy as np
import pandas as pd
import scipy.stats, scipy.spatial.transform
import matplotlib.pyplot as plt
import seaborn as sns
import cm_functions as cmf


main_data_path = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/pickle/bh_perturb/"
data_files = [
    "NGCa0524_bhperturb.pickle", "NGCa2986_bhperturb.pickle", "NGCa3348_bhperturb.pickle", "NGCa3607_bhperturb.pickle", "NGCa4291_bhperturb.pickle"
]

#Apply a rotation to the data?
apply_rotation = False

#mask the data?
lower_time = 4

#set up dataframe for seaborn use
#whilst iterating, also test goodness of fit to normal distribution
fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].set_title("Position")
ax[1].set_title("Velocity")
cols = cmf.plotting.mplColours()
name_list = []
Astat_pos = np.full((len(data_files), 3), np.nan, dtype=float)
Astat_vel = np.full_like(Astat_pos, np.nan)
for i, datafile in enumerate(data_files):
    data_dict = cmf.utils.load_data(os.path.join(main_data_path, datafile))
    if apply_rotation:
        rotation_quat = scipy.spatial.transform.Rotation.random()
        print(rotation_quat.as_quat())
        data_dict["diff_x"] = rotation_quat.apply(data_dict["diff_x"])
        data_dict["diff_v"] = rotation_quat.apply(data_dict["diff_v"])
    time_mask = data_dict["times"] > lower_time
    names = [data_dict["galaxy_name"]] * len(data_dict["diff_x"][time_mask,0])
    tempdict = {"x":data_dict["diff_x"][time_mask,0], 
                "y":data_dict["diff_x"][time_mask,1], 
                "z":data_dict["diff_x"][time_mask,2], 
                "vx":data_dict["diff_v"][time_mask,0], 
                "vy":data_dict["diff_v"][time_mask,1], 
                "vz":data_dict["diff_v"][time_mask,2], "name":names}
    #perform the Anderson-Darling test for normality
    print("Anderson Test for {}".format(data_dict["galaxy_name"]))
    name_list.append(data_dict["galaxy_name"])
    for Astatarr, vals in zip((Astat_pos, Astat_vel), (data_dict["diff_x"], data_dict["diff_v"])):
        for j in range(3):
            Astatarr[i,j], critval, siglev = scipy.stats.anderson(vals[time_mask,j], dist="norm")
    if i==0:
        dataframe = pd.DataFrame.from_dict(tempdict)
    else:
        dataframe2 = pd.DataFrame.from_dict(tempdict)
        dataframe = dataframe.append(dataframe2, ignore_index=True)
for i, c in enumerate(critval):
    ax[0].axhline(c, c=cols[i], label="{}%".format(siglev[i]))
    ax[1].axhline(c, c=cols[i])
for i, arr in enumerate((Astat_pos, Astat_vel)):
    for j, (marker, label) in enumerate(zip(("o", "s", "^"), ("x", "y", "z"))):
        ax[i].scatter(name_list, arr[:,j], c="k", label=r"$A^{{*2}}_{}$".format(label), zorder=10, marker=marker)
    plt.sca(ax[i])
    plt.xticks(rotation=45)
ax[0].legend()
if apply_rotation:
    suffix="random"
else:
    suffix="iso"
fig.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/anderson_{}.png".format(suffix)))

#plot the displacement and velocity magnitudes, with a kernel-density estimate
px = sns.jointplot(data=dataframe, x="x", y="vx", hue="name")
px.set_axis_labels("x Displacement [kpc]", "x-Velocity [km/s]")
px.figure.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/x_{}.png".format(suffix)))
py = sns.jointplot(data=dataframe, x="y", y="vy", hue="name")
py.set_axis_labels("y Displacement [kpc]", "y-Velocity [km/s]")
py.figure.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/y_{}.png".format(suffix)))
pz = sns.jointplot(data=dataframe, x="z", y="vz", hue="name")
pz.set_axis_labels("z Displacement [kpc]", "z-Velocity [km/s]")
pz.figure.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/z_{}.png".format(suffix)))
plt.show()