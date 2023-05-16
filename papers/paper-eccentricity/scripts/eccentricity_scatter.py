import numpy as np
import matplotlib.pyplot as plt
import pickle
import cm_functions as cmf
import figure_config


#------------------------------------------------------------
# plot of convergence in sigma_e across resolutions
#------------------------------------------------------------

# read in data
data_e090 = cmf.utils.load_data(figure_config.data_path("nasim_scatter_e0-0.900.pickle"))
data_e099 = cmf.utils.load_data(figure_config.data_path("nasim_scatter_e0-0.990.pickle"))

# set up the figure
fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$M_\bullet/m_\star$")
ax.set_xscale("log")
ax.set_ylabel(r"$\sigma_e$")
ax.set_yscale("log")
marker_kwargs = {"edgecolor":"k", "lw":0.5}
cmapper, sm = cmf.plotting.create_normed_colours(5e-3, 0.2, cmap="BuPu", normalisation="LogNorm")
markers = cmf.plotting.mplChars()


# plot data
for i, (d, lab) in enumerate(zip((data_e090, data_e099), (r"$e_0=0.90$", r"$e_0=0.99$"))):
    sc = ax.scatter(d["mass_res"], d["sigma_e"], label=f"{lab}", c=cmapper(1-d["e_ini"]), marker=markers[0], zorder=10, **marker_kwargs)
    ax.scatter(d["mass_res"], d["sigma_e_cut"], label=f"{lab} ($<{d['sigma_e_cut_threshold']:.1f}\\sigma$)", c=cmapper(1-d["e_ini"]), marker=markers[1], s=sc.get_sizes()*1.5, zorder=1, **marker_kwargs)

# add the sqrt{resolution} scaling line
mass_res_seq = np.geomspace(2e3, 5e4, 10)
ax.plot(mass_res_seq, 3*mass_res_seq**-0.5, c="k", label=r"$\propto 1/\sqrt{M_\bullet/m_\star}$")

# final touch ups
cmf.plotting.nice_log10_scale(ax, "xy")
ax.legend(fontsize="small")

# save
cmf.plotting.savefig(figure_config.fig_path("convergence.pdf"), force_ext=True)
plt.close(fig)


#------------------------------------------------------------
# plot of sigma_e across initial eccentricities
#------------------------------------------------------------

# read in data
data_2M = cmf.utils.load_data("nasim_scatter_res-2.0e+04.pickle")

# set up the figure
fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$e_0$")
ax.set_ylabel(r"$\sigma_e$")
ax.set_yscale("log")

ax.scatter(data_2M["e_ini"], data_2M["sigma_e"], **marker_kwargs)
cmf.plotting.nice_log10_scale(ax, "y")
cmf.plotting.savefig(figure_config.fig_path("eccentricities.pdf"), force_ext=True)
plt.close(fig)