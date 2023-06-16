import numpy as np
import matplotlib.pyplot as plt
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

# set mass resolution - N_star mapping, 
# knowing that the BH mass is 1e8 Msol and the galaxy has stellar mass 
# 1e10 Msol
N_star = data_e090["mass_res"] / 1e8 * 1e10 * 2
# set up the twin axis
ax_twin = cmf.plotting.twin_axes_from_samples(ax, data_e090["mass_res"], N_star, log=False)
ax_twin.set_xlabel(r"$N_{\star,\mathrm{tot}}$")

# plot data
marker_list = figure_config.marker_cycle.by_key()["marker"]
col_list = figure_config.color_cycle_shuffled.by_key()["color"]
for i, (d, lab, marker, col) in enumerate(zip((data_e090, data_e099), (r"$e_0=0.90$", r"$e_0=0.99$"), marker_list, col_list)):
    l = ax.plot(d["mass_res"], d["sigma_e"], label=f"{lab}", zorder=1, ls="", marker=marker, mec=col, mfc="None")
    ax.plot(d["mass_res"], d["sigma_e_cut"], label=f"{lab} ($<{d['sigma_e_cut_threshold']:.1f}\\sigma$)", zorder=1, ls="", marker=marker, mfc=col)

# add the sqrt{resolution} scaling line
mass_res_seq = np.geomspace(2e3, 5e4, 10)
ax.plot(mass_res_seq, 3*mass_res_seq**-0.5, c="k", label=r"$\propto 1/\sqrt{N_{\star,\mathrm{tot}}}$")

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
data_2M = cmf.utils.load_data(figure_config.data_path("nasim_scatter_res-2.0e+04.pickle"))

# set up the figure
fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$e_0$")
ax.set_ylabel(r"$\sigma_e$")
ax.set_yscale("log")

ax.scatter(data_2M["e_ini"], data_2M["sigma_e"])
cmf.plotting.nice_log10_scale(ax, "y")