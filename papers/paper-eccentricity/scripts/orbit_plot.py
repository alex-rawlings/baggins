import os
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import figure_config



angle_data = cmf.utils.load_data("data/deflection_angles_e0-0.990.pickle")

# print some info if desired to help choose sims to plot
if False:
    for i, (res, ang) in enumerate(zip(angle_data["mass_res"], angle_data["thetas"])):
        print(f"{i:02d}: res: {res:.1e}, angle: {ang:.1f}")
    raise RuntimeError


angles = [
    angle_data["thetas"][18],
    angle_data["thetas"][24]
]

main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/"



# XXX MUST MATCH THESE WITH THE CORRESPONDING DATA ABOVE, NOT DONE AUTO
data_dirs = [
    os.path.join(main_path, "500K/D_500K_c-D_500K_d-3.720-0.028"),
    os.path.join(main_path, "1M/D_1M_b-D_1M_b-3.720-0.028"),
]

# turn off for fast plotting, line will be a single colour
use_gradient_line = True

# some shortcuts
min_time_colour = 22
idx_stride = 10
A_idx = 3140
B_idx = A_idx + 35


# set up figure
fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$x/\mathrm{kpc}$")
ax.set_ylabel(r"$z/\mathrm{kpc}$")
axins1 = ax.inset_axes([0.05, 0.4, 0.4, 0.4])
axins2 = ax.inset_axes([0.6, 0.05, 0.4, 0.4])

glp = cmf.plotting.GradientLinePlot(ax=ax, cmap="cividis")

for i, (dd, axins, ang) in enumerate(zip(data_dirs, (axins1, axins2), angles)):
    ketjufile = cmf.utils.get_ketjubhs_in_dir(dd)[0]
    bh1, bh2 = cmf.analysis.get_binary_before_bound(ketjufile)
    bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)
    # ensure the plotted BH starts in the upper right corner
    if bh1.x[0,2] > 0:
        bh = bh1
    else:
        bh = bh2
    x = bh.x[::idx_stride,0] / cmf.general.units.kpc
    y = bh.x[::idx_stride,2] / cmf.general.units.kpc
    t = bh.t[::idx_stride] / cmf.general.units.Myr
    if use_gradient_line:
        glp.add_data(x, y, c=t)
    else:
        ax.plot(x,y)
        axins.plot(x,y)
    # annotate points just before and just after the first hard scatter
    axins.annotate(
                "A", 
                (x[A_idx], y[A_idx]),
                xytext=(-13, 10),
                textcoords="offset points",
                arrowprops={"arrowstyle":"wedge", "fc":"k"},
                horizontalalignment="right",
                verticalalignment="bottom"
                )
    axins.annotate(
                "B", 
                (x[B_idx], y[B_idx]),
                xytext=(5, -20),
                textcoords="offset points",
                arrowprops={"arrowstyle":"wedge", "fc":"k"},
                horizontalalignment="left",
                verticalalignment="top"
                )

    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title(f"$\\theta_\mathrm{{defl}}={ang:.1f}\degree$", fontsize="small")


glp.plot(logcolour=False, vmin=min_time_colour)
glp.plot_single_series(0, logcolour=False, ax=axins1, vmin=min_time_colour)
glp.plot_single_series(1, logcolour=False, ax=axins2, vmin=min_time_colour)
glp.add_cbar(ax, label=r"$t/\mathrm{Myr}$", extend="min")

axins1.set_xlim(-0.10, 0.05)
axins1.set_ylim(-0.10, 0.03)
axins2.set_xlim(-0.07, 0.03)
axins2.set_ylim(-0.13, 0.05)
ax.set_xlim(ax.get_xlim() * np.array([1, 2]))

for axins in (axins1, axins2):
    ax.indicate_inset_zoom(axins, edgecolor="k")

ax.text(0.7, 0.9, r"$e_0=0.99$", transform=ax.transAxes)

if use_gradient_line:
    cmf.plotting.savefig(figure_config.fig_path("orbit.pdf"), force_ext=True)
plt.show()