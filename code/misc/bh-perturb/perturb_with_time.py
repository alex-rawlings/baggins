import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs


def gradient_line(ax, x, y, colours, **plot_kwargs):
    #ax.plot(x[0], y[0], ls='none', color=colours[0], marker="X", **plot_kwargs)
    for xs, ys, c in zip(zip(x[:-1], x[1:]), zip(y[:-1], y[1:]), colours[:-1]):
        ax.plot(xs, ys, color=c, markevery=[-1], **plot_kwargs)
    

main_data_path = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/pickle/bh_perturb/"
data_files = [
    "NGCa0524_bhperturb.pickle", "NGCa2986_bhperturb.pickle", "NGCa3348_bhperturb.pickle", "NGCa3607_bhperturb.pickle", "NGCa4291_bhperturb.pickle"
]


fig, ax = plt.subplots(1,2,sharex="all")
markers = bgs.plotting.mplChars()
cmap = plt.cm.viridis
for i, datafile in enumerate(data_files):
    data_dict = bgs.utils.load_data(os.path.join(main_data_path, datafile))
    displacement = bgs.mathematics.radial_separation(data_dict["diff_x"])
    vel_mag = bgs.mathematics.radial_separation(data_dict["diff_v"])

    if i==0:
        gradplot_pos = bgs.plotting.GradientScatterPlot(ax[0], data_dict["times"], data_dict["stellar_density"], displacement, label=data_dict["galaxy_name"], marker=markers[i])
        gradplot_vel = bgs.plotting.GradientLinePlot(ax[1], data_dict["times"], data_dict["stellar_density"], vel_mag, label=data_dict["galaxy_name"], marker=markers[i])
    else:
        gradplot_pos.add_data(data_dict["times"], data_dict["stellar_density"], displacement, label=data_dict["galaxy_name"], marker=markers[i])
        gradplot_vel.add_data(data_dict["times"], data_dict["stellar_density"], vel_mag, label=data_dict["galaxy_name"], marker=markers[i])

for axi in (gradplot_pos.ax, gradplot_vel.ax):
    axi.set_xlabel("t/Gyr")
    axi.set_ylabel(r"$\bar{\rho} / (\mathrm{M}_\odot / \mathrm{kpc}^3)$")
gradplot_pos.plot()
gradplot_pos.add_legend()
gradplot_pos.add_cbar(label="Displacement/kpc")
gradplot_vel.plot()
gradplot_vel.add_cbar(label="Velocity/(km/s)")
plt.show()