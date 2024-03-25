import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config

#------------------------------------------------------------
# plot of convergence in sigma_e across resolutions
#------------------------------------------------------------

# read in data
data_e090 = bgs.utils.load_data(figure_config.data_path("nasim_scatter_e0-0.900.pickle"))
data_e099 = bgs.utils.load_data(figure_config.data_path("nasim_scatter_e0-0.990.pickle"))

# nasim 2020 data. from Fig. 8
nasim = dict(
    N_half = np.array([62654.24645191446, 123482.87969931985, 248057.20042674404, 1009270.5735803195]),
    sigma_e = np.array([0.16478219227026006, 0.08915510513943252, 0.06637728132363757, 0.0327192313547047])
)
# convert nasim x values
nasim["mass_res"] = 1e8/(1e10/(nasim["N_half"]*2))


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
ax_twin = bgs.plotting.twin_axes_from_samples(ax, data_e090["mass_res"], N_star, log=False)
ax_twin.set_xlabel(r"$N_{\star,\mathrm{tot}}$")

# plot data
marker_list = figure_config.marker_cycle.by_key()["marker"]
col_list = figure_config.color_cycle_shuffled.by_key()["color"]
for i, (d, lab, marker, col) in enumerate(zip((data_e090, data_e099, nasim), (r"$e_0=0.90$", r"$e_0=0.99$", r"$\mathrm{Nasim+20}$"), marker_list, col_list)):
    l = ax.plot(d["mass_res"], d["sigma_e"], label=f"{lab}", zorder=1, ls="", marker=marker, c=col)

# add the sqrt{resolution} scaling line
mass_res_seq = np.geomspace(1.5e3, 3e4, 10)
ax.plot(mass_res_seq, 1.2*mass_res_seq**-0.5, c="k", label=r"$\propto 1/\sqrt{N_{\star,\mathrm{tot}}}$")

# final touch ups
bgs.plotting.nice_log10_scale(ax, "xy")
ax.set_xlim(7e2, ax.get_xlim()[1])
ax.legend(fontsize="small", loc="lower left")

# save
bgs.plotting.savefig(figure_config.fig_path("convergence.pdf"), force_ext=True)