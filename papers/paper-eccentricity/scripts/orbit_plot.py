import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cm_functions as cmf
import figure_config



print_data = False
# turn off for fast plotting, line will be a single colour
use_gradient_line = True

idx_stride = 10
min_time_colour = -2
max_time_colour = -min_time_colour
sizebar_kwargs = {"width":3, "remove_ticks":False, "unitconvert":"kilo2base", "textsize":"small", "fmt":".0f"}
gradline_kwargs = {"ls":"-", "capstyle":"round"}

# set up figure
fig, ax = plt.subplots(1,2, figsize=(6,3), sharex="all", sharey="all")
ax[0].set_ylabel(r"$z/\mathrm{kpc}$")

# combine the two geometries into one figure with two panels
for j in range(2):
    print(f"Making subplot {j}")
    bound_idx = []
    bound_idx_strided = []
    if j==0:
        angle_data = cmf.utils.load_data("data/deflection_angles_e0-0.900.pickle")

        # print some info if desired to help choose sims to plot
        if print_data:
            for i, (res, ang) in enumerate(zip(angle_data["mass_res"], angle_data["thetas"])):
                print(f"{i:02d}: res: {res:.1e}, angle: {ang:.1f}")
            print("------------")

        angles = [
            angle_data["thetas"][1],
            angle_data["thetas"][6]
        ]

        main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/"
        fig_prefix = "e90"
        label = r"$e_0=0.90$"

        # XXX MUST MATCH THESE WITH THE CORRESPONDING DATA ABOVE, NOT DONE AUTO
        data_dirs = [
            os.path.join(main_path, "100K/D_100K_a-D_100K_c-3.720-0.279"),
            os.path.join(main_path, "100K/D_100K_b-D_100K_e-3.720-0.279"),
        ]

        # some shortcuts
        A_idx = [4550, 4525]
        B_idx = [A_idx[0]+26, A_idx[1]+16]
        extra_bound_bit = [1000, 1000]
    else:
        angle_data = cmf.utils.load_data("data/deflection_angles_e0-0.990.pickle")

        # print some info if desired to help choose sims to plot
        if print_data:
            for i, (res, ang) in enumerate(zip(angle_data["mass_res"], angle_data["thetas"])):
                print(f"{i:02d}: res: {res:.1e}, angle: {ang:.1f}")
            raise RuntimeError

        angles = [
            angle_data["thetas"][4],
            angle_data["thetas"][7]
        ]

        main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/"
        fig_prefix = "e99"
        label = r"$e_0=0.99$"

        # XXX MUST MATCH THESE WITH THE CORRESPONDING DATA ABOVE, NOT DONE AUTO
        data_dirs = [
            os.path.join(main_path, "100K/D_100K_b-D_100K_c-3.720-0.028"),
            os.path.join(main_path, "100K/D_100K_c-D_100K_d-3.720-0.028"),
        ]

        # some shortcuts
        A_idx = [3135, 3135]
        B_idx = [A_idx[0]+40, A_idx[1]+64]
        extra_bound_bit = [1500, 800]


    ax[j].set_xlabel(r"$x/\mathrm{kpc}$")
    axins1 = ax[j].inset_axes(
                            [0.05, 0.4, 0.38, 0.4] if j==0 else 
                            [0.05, 0.4, 0.38, 0.4]
                            )
    axins2 = ax[j].inset_axes(
                            [0.65, 0.05, 0.35, 0.4] if j==0 else
                            [0.55, 0.05, 0.4, 0.4]
                            )
    # keep axis limits fixed for comparison between orbits
    ax[j].set_xlim(-0.7, 1.2)
    ax[j].set_ylim(-0.5, 2)
    for axi in (ax[j], axins1, axins2): axi.set_prop_cycle(figure_config.color_cycle)

    glp = cmf.plotting.GradientLinePlot(ax=ax[j], cmap="custom_diverging")
    glp.marker_kwargs = {"ec":"w", "lw":0.2}

    for i, (dd, axins, ang) in enumerate(zip(data_dirs, (axins1, axins2), angles)):
        ketjufile = cmf.utils.get_ketjubhs_in_dir(dd)[0]
        bh1, bh2, *_ = cmf.analysis.get_bh_particles(ketjufile)
        bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)
        bh1u, bh2u, *_ = cmf.analysis.get_binary_before_bound(ketjufile)
        bound_idx.append(len(bh1u))
        bound_idx_strided.append(int(np.floor(bound_idx[i]/idx_stride)))

        # ensure the plotted BH starts in the upper right corner
        if bh1.x[0,2] > 0:
            bh = bh1
        else:
            bh = bh2
        bh = bh[:bound_idx[i]+extra_bound_bit[i]]
        x = bh.x[::idx_stride,0] / cmf.general.units.kpc
        y = bh.x[::idx_stride,2] / cmf.general.units.kpc
        t = bh.t[::idx_stride] / cmf.general.units.Myr
        t = t - t[bound_idx_strided[i]]
        if use_gradient_line:
            glp.add_data(x, y, c=t)
        else:
            ax[j].plot(x,y, markevery=[bound_idx_strided[i]], marker="o")
            axins.plot(x,y, markevery=[bound_idx_strided[i]], marker="o")
        # annotate points just before and just after the first hard scatter

        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title(f"$\\theta_\mathrm{{defl}}={ang:.1f}\degree$", fontsize="small")

    if use_gradient_line:
        glp.plot(logcolour=False, vmin=min_time_colour, vmax=max_time_colour, marker_idx=bound_idx_strided, **gradline_kwargs)
        lc1 = glp.plot_single_series(0, logcolour=False, ax=axins1, vmin=min_time_colour, vmax=max_time_colour, marker_idx=bound_idx_strided[0], **gradline_kwargs)
        lc2 = glp.plot_single_series(1, logcolour=False, ax=axins2, vmin=min_time_colour, vmax=max_time_colour, marker_idx=bound_idx_strided[1], **gradline_kwargs)
        for i, axins in enumerate((axins1, axins2)):
            glp.draw_arrow_on_series(axins, i, A_idx[i])
            glp.draw_arrow_on_series(axins, i, B_idx[i])
        if j==1:
            cbar = glp.add_cbar(ax[j], label=r"$t'/\mathrm{Myr}$", extend="both")
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if j==0:
        axins1.set_xlim(-0.015, 0.033)
        axins1.set_ylim(-0.03, 0.04)
        for axins in (axins1, axins2):
            cmf.plotting.draw_sizebar(axins, 0.01, "pc", location="lower right", **sizebar_kwargs)
    else:
        axins1.set_xlim(-0.06, 0.05)
        axins1.set_ylim(-0.08, 0.03)
        for axins in (axins1, axins2):
            cmf.plotting.draw_sizebar(axins, 0.01, "pc", location="lower right", **sizebar_kwargs)
    axins2.set_xlim(*axins1.get_xlim())
    axins2.set_ylim(*axins1.get_ylim())
    axins1.set_aspect("equal")
    axins2.set_aspect("equal")

    for axins in (axins1, axins2):
        ax[j].indicate_inset_zoom(axins, edgecolor="k")

    ax[j].set_title(label)

if use_gradient_line:
    cmf.plotting.savefig(figure_config.fig_path(f"orbit.pdf"), force_ext=True)
else:
    plt.show()
