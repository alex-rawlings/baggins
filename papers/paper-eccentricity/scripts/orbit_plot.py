import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf
import figure_config



use_e90_data = True
print_data = False
# turn off for fast plotting, line will be a single colour
use_gradient_line = True

if use_e90_data:
    angle_data = cmf.utils.load_data("data/deflection_angles_e0-0.900.pickle")

    # print some info if desired to help choose sims to plot
    if print_data:
        for i, (res, ang) in enumerate(zip(angle_data["mass_res"], angle_data["thetas"])):
            print(f"{i:02d}: res: {res:.1e}, angle: {ang:.1f}")
        raise RuntimeError

    angles = [
        angle_data["thetas"][16],
        angle_data["thetas"][28]
    ]

    main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/"
    fig_prefix = "e90"
    label = r"$e_0=0.90$"

    # XXX MUST MATCH THESE WITH THE CORRESPONDING DATA ABOVE, NOT DONE AUTO
    data_dirs = [
        os.path.join(main_path, "500K/D_500K_b-D_500K_d-3.720-0.279"),
        os.path.join(main_path, "1M/D_1M_c-D_1M_d-3.720-0.279"),
    ]

    # some shortcuts
    min_time_colour = 31
    idx_stride = 10
    A_idx = [4500, 4280]
    B_idx = [A_idx[0]+35, A_idx[1]+30]
else:
    angle_data = cmf.utils.load_data("data/deflection_angles_e0-0.990.pickle")

    # print some info if desired to help choose sims to plot
    if print_data:
        for i, (res, ang) in enumerate(zip(angle_data["mass_res"], angle_data["thetas"])):
            print(f"{i:02d}: res: {res:.1e}, angle: {ang:.1f}")
        raise RuntimeError

    angles = [
        angle_data["thetas"][18],
        angle_data["thetas"][24]
    ]

    main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/"
    fig_prefix = "e99"
    label = r"$e_0=0.99$"

    # XXX MUST MATCH THESE WITH THE CORRESPONDING DATA ABOVE, NOT DONE AUTO
    data_dirs = [
        os.path.join(main_path, "500K/D_500K_c-D_500K_d-3.720-0.028"),
        os.path.join(main_path, "1M/D_1M_b-D_1M_b-3.720-0.028"),
    ]

    # some shortcuts
    min_time_colour = 22
    idx_stride = 10
    A_idx = [3140, 3140]
    B_idx = [A_idx[0]+35, A_idx[1]+35]


# set up figure
fig, ax = plt.subplots(1,1)
fig.set_figwidth(fig.get_figwidth()*1.3)
ax.set_xlabel(r"$x/\mathrm{kpc}$")
ax.set_ylabel(r"$z/\mathrm{kpc}$")
axins1 = ax.inset_axes([0.05, 0.4, 0.3, 0.3])
axins2 = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
# keep axis limits fixed for comparison between orbits
ax.set_xlim(-0.7, 1.2)
ax.set_ylim(-0.5, 2)

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
    if use_e90_data:
        axins.annotate(
                "A", 
                (x[A_idx[i]], y[A_idx[i]]),
                xytext=(-13, -10),
                textcoords="offset points",
                arrowprops={"arrowstyle":"wedge", "fc":"k"},
                horizontalalignment="right",
                verticalalignment="bottom"
                )
        axins.annotate(
                    "B", 
                    (x[B_idx[i]], y[B_idx[i]]),
                    xytext=(10, -10),
                    textcoords="offset points",
                    arrowprops={"arrowstyle":"wedge", "fc":"k"},
                    horizontalalignment="left",
                    verticalalignment="top"
                )
    else:
        axins.annotate(
                    "A", 
                    (x[A_idx[i]], y[A_idx[i]]),
                    xytext=(-13, 10),
                    textcoords="offset points",
                    arrowprops={"arrowstyle":"wedge", "fc":"k"},
                    horizontalalignment="right",
                    verticalalignment="bottom"
                    )
        axins.annotate(
                    "B", 
                    (x[B_idx[i]], y[B_idx[i]]),
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

if use_gradient_line:
    glp.plot(logcolour=False, vmin=min_time_colour)
    glp.plot_single_series(0, logcolour=False, ax=axins1, vmin=min_time_colour)
    glp.plot_single_series(1, logcolour=False, ax=axins2, vmin=min_time_colour)
    glp.add_cbar(ax, label=r"$t/\mathrm{Myr}$", extend="min")

if use_e90_data:
    axins1.set_xlim(-0.02, 0.05)
    axins1.set_ylim(-0.08, 0.09)
    axins2.set_xlim(-0.02, 0.08)
    axins2.set_ylim(-0.08, 0.03)
else:
    axins1.set_xlim(-0.10, 0.05)
    axins1.set_ylim(-0.10, 0.03)
    axins2.set_xlim(-0.07, 0.03)
    axins2.set_ylim(-0.13, 0.05)

for axins in (axins1, axins2):
    ax.indicate_inset_zoom(axins, edgecolor="k")

ax.annotate(label, (0.7,0.9), xycoords=ax.transAxes, bbox={"boxstyle":"square", "fc":"w", "ec":"k"})

if use_gradient_line:
    cmf.plotting.savefig(figure_config.fig_path(f"orbit-{fig_prefix}.pdf"), force_ext=True)

