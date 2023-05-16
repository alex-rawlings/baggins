import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf
import figure_config

main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/2M"
data_dirs = [
    os.path.join(main_path, "D_2M_a-D_2M_a-3.720-0.028"),
    os.path.join(main_path, "D_2M_a-D_2M_b-3.720-0.028"),
    os.path.join(main_path, "D_2M_a-D_2M_c-3.720-0.028"),
]

# turn off for fast plotting, line will be a single colour
use_gradient_line = True

# some shortcuts
kpc = cmf.general.units.kpc
Myr = cmf.general.units.Myr
min_time_colour = 22
A_idx = 31400
B_idx = A_idx + 400


# set up figure
fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$x/\mathrm{kpc}$")
ax.set_ylabel(r"$z/\mathrm{kpc}$")
axins = ax.inset_axes([0.1, 0.4, 0.5, 0.5])


for dd in data_dirs:
    ketjufile = cmf.utils.get_ketjubhs_in_dir(dd)[0]
    bh1, bh2 = cmf.analysis.get_binary_before_bound(ketjufile)
    bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)
    # ensure the plotted BH starts in the upper right corner
    if bh1.x[0,2] > 0:
        bh = bh1
    else:
        bh = bh2
    if use_gradient_line:
        glp = cmf.plotting.GradientLinePlot(ax=ax, cmap="cividis")
        glp.add_data(bh.x[1:,0]/kpc, bh.x[1:,2]/kpc, c=bh.t[1:]/Myr)
        glp.plot(logcolour=False, vmin=min_time_colour)
        glp.plot(logcolour=False, ax=axins, vmin=min_time_colour)
        glp.add_cbar(ax, label=r"$t/\mathrm{Myr}$", extend="min")
    else:
        ax.plot(bh.x[1:,0]/kpc, bh.x[1:,2]/kpc)
        axins.plot(bh.x[1:,0]/kpc, bh.x[1:,2]/kpc)
    # annotate points just before and just after the first hard scatter
    axins.annotate(
                "A", 
                (bh.x[A_idx,0]/kpc, bh.x[A_idx,2]/kpc),
                xytext=(-15, 25),
                textcoords="offset points",
                arrowprops={"arrowstyle":"wedge", "fc":"k"},
                horizontalalignment="right",
                verticalalignment="bottom"
                )
    axins.annotate(
                "B", 
                (bh.x[B_idx,0]/kpc, bh.x[B_idx,2]/kpc),
                xytext=(12, -25),
                textcoords="offset points",
                arrowprops={"arrowstyle":"wedge", "fc":"k"},
                horizontalalignment="left",
                verticalalignment="top"
                )
    break

axins.set_xlim(-0.1, 0.05)
axins.set_ylim(-0.10, 0.03)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="k")

if use_gradient_line:
    cmf.plotting.savefig(figure_config.fig_path("orbit.pdf"), force_ext=True)
plt.show()