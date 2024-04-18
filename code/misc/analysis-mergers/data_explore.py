import os.path
import numpy as np
import scipy.stats, scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import baggins as bgs
import pygad

sclw = 0.5

def pygad_for_seaborn(a):
    return a.view(np.ndarray).flatten()[0]

def data_helper(a, c, nm, dkey="estimate"):
    if a == "merger_name":
        x = nm[5:-9]
        label = "family"
    else:
        x = getattr(c, a)
        label = a
        if isinstance(x, dict):
            x = x[dkey]
            label += "_{}".format(dkey)
        if isinstance(x, pygad.UnitArr):
            x = pygad_for_seaborn(x)
    return x, label

def extract_and_plot_point(attrs, xdkey="estimate", ydkey="estimate", zdkey="merged", categorical_y=False, earlyreturn=False):
    rawdat = []
    for i, c in enumerate(cubes):
        print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
        cdc = bgs.analysis.ChildSimData.load_from_file(c)

        if not cdc.relaxed_remnant_flag: continue

        myname = c.split("/")[-1]
        
        x, xlabel = data_helper(attrs[0], cdc, myname, dkey=xdkey)
        y, ylabel = data_helper(attrs[1], cdc, myname, dkey=ydkey)
        z, zlabel = data_helper(attrs[2], cdc, myname, dkey=zdkey)

        rawdat.append([
                        myname,
                        x,
                        y,
                        z,
                        cdc.relaxed_remnant_flag
        ])
    print("\nComplete")

    df = pd.DataFrame(rawdat, columns=["names", xlabel, ylabel, zlabel, "relaxed"])
    #pd.set_option("display.max_rows", 500)
    print(df)
    if not categorical_y:
        try:
            cmap = sns.color_palette("viridis", as_cmap=True)
            p = sns.jointplot(data=df, x=xlabel, y=ylabel, hue=zlabel, palette=cmap)
        except TypeError:
            p = sns.jointplot(data=df, x=xlabel, y=ylabel, hue=zlabel)
    else:
        alpha = [1, 0.4]
        cols = bgs.plotting.mplColours()
        for i, zi in enumerate(np.unique(df.loc[:,zlabel])):
            mask = df.loc[:,zlabel] == zi
            for j, b in enumerate((True, False)):
                #mask = mask & (df.loc[:,"relaxed"]==b)
                mask = np.logical_and(mask, df.loc[:,"relaxed"]==b)
                p = plt.scatter(df.loc[mask,xlabel], df.loc[mask,ylabel], label=(zi if j<1 else ""), linewidths=sclw, edgecolors="k", alpha=alpha[j], c=cols[i])
        plt.legend(title=zlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if earlyreturn:
            return df
    plt.savefig(os.path.join(bgs.FIGDIR, "analysis-explore/{}-{}.png".format(xlabel, ylabel)))
    return p, df
    #plt.show()
    #quit()


if __name__ == "__main__":
    cubes = bgs.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes", recursive=True)
    num_sims = len(cubes)
    call = 1
    

    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["r_bound_ecc", "r_infl_ecc", "merger_name"]
        extract_and_plot_point(attrs)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["r_bound_ecc", "r_hard_ecc", "merger_name"]
        extract_and_plot_point(attrs)
    
    if True:
        print("Call {}".format(call))
        call += 1
        attrs = ["r_hard_time", "K", "merger_name"]
        extract_and_plot_point(attrs)
        plt.show()
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["r_infl_ecc", "r_hard_ecc", "merger_name"]
        extract_and_plot_point(attrs)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["parent_quantities", "r_bound_ecc", "binary_merger_remnant"]
        extract_and_plot_point(attrs, xdkey="initial_e")

    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["relaxed_effective_radius", "virial_info", "merger_name"]
        _,df = extract_and_plot_point(attrs, ydkey="radius")
        pd.set_option("display.max_rows", 500)
        print(df.loc[:,["names", "virial_info_radius"]])
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["virial_info", "relaxed_half_mass_radius", "merger_name"]
        jp,_ = extract_and_plot_point(attrs, xdkey="radius")
        rvir = np.linspace(300, 800, 1000)
        jp.ax_joint.plot(rvir, bgs.literature.Kratsov13(rvir))
        plt.show()
    
    if False:
       # misgeld = pd.read_table("../../initialise_scripts/literature_data/misgeld_11.dat", sep="|", header=0, skiprows=[1,3], skipinitialspace=True, comment="#")
        print("Call {}".format(call))
        call += 1
        attrs = ["virial_info", "total_stellar_mass", "merger_name"]
        df = extract_and_plot_point(attrs, xdkey="mass", categorical_y=True)
        plt.xlim(1e13, 2e14)
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(os.path.join(bgs.FIGDIR, "analysis-explore/{}-{}.png".format(attrs[0]+"_mass", attrs[1])))
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["relaxed_effective_radius", "relaxed_inner_DM_fraction", "merger_name"]
        extract_and_plot_point(attrs)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_lifetime_timescale", "merger_name", "binary_merger_remnant"]
        extract_and_plot_point(attrs, categorical_y=True)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_lifetime_timescale", "binary_spin_flip", "binary_merger_remnant"]
        _,df = extract_and_plot_point(attrs=attrs, categorical_y=True)
        mask = df.loc[:,"binary_merger_remnant_merged"] & df.loc[:,"binary_spin_flip"]
        num_flips = len(df.loc[df.loc[:,"binary_spin_flip"], "binary_spin_flip"])
        num_flips_merged = len(df.loc[mask, "binary_spin_flip"])
        print("{}/{} mergers with spin flips merged.".format(num_flips_merged, num_flips))
        mask = df.loc[:,"binary_merger_remnant_merged"] & ~df.loc[:,"binary_spin_flip"]
        num_no_flips = len(df.loc[~df.loc[:,"binary_spin_flip"],"binary_spin_flip"])
        num_no_flips_merged = len(df.loc[mask, "binary_spin_flip"])
        print("{}/{} mergers without spin flips merged.".format(num_no_flips_merged, num_no_flips))
    
    if False:
        print("Call {}".format(call))
        call += 1
        cols = bgs.plotting.mplColours()

        fig, ax = plt.subplots(1,1)
        ax.axhline(np.pi/2, c="k", ls="--")
        ax.set_ylim(0, np.pi)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\theta$")

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            cdc = bgs.analysis.ChildSimData.load_from_file(c)
            #marker = "o" if cdc.binary_merger_remnant["merged"] else ""
            alpha = 0.9  if cdc.binary_merger_remnant["merged"] else 0.3
            ax.plot(cdc.snapshot_times, cdc.ang_mom_diff_angle, c=cols[int(cdc.binary_spin_flip)], alpha=alpha)
        plt.show()

    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_lifetime_timescale", "parent_quantities", "binary_spin_flip"]
        extract_and_plot_point(attrs, ydkey="initial_e", zdkey=None)

    if False:
        print("Call {}".format(call))
        call += 1

        fig, ax = plt.subplots(1,1)
        #colour lines by their age
        cmapper, sm = bgs.plotting.create_normed_colours(0, 13.8e3)

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            cdc = bgs.analysis.ChildSimData.load_from_file(c)
            age = cdc.parent_quantities["perturb_time"] + cdc.binary_lifetime_timescale
            ax.loglog(cdc.radial_bin_centres["stars"], cdc.relaxed_density_profile["stars"], c=cmapper(age))
        plt.colorbar(sm)
        plt.show()
    
    if False:
        for c in cubes:
            cdc = bgs.analysis.ChildSimData.load_from_file(c)
            myname = c.split("/")[-1]
            if cdc.relaxed_remnant_flag: break
        eb = [np.mean(e) for e in cdc.binding_energy_bins]
        plt.plot(eb, cdc.relaxed_triaxiality_parameters["ba"], label="b/a")
        plt.plot(eb, cdc.relaxed_triaxiality_parameters["ca"], label="c/a")
        plt.xlabel(r"Binding Energy [M$_\odot$km$^2$s$^{-2}$]")
        plt.ylim(0, 1)
        plt.legend()
        plt.title(myname)
        plt.savefig(os.path.join(bgs.FIGDIR, f"analysis-explore/{myname}-triaxial.png"))
    
    if False:
        print("Call {}".format(call))
        call += 1

        fig, ax = plt.subplots(2,1, sharex="all", sharey="all")
        gradplotM = bgs.plotting.GradientLinePlot(ax[0], cmap="plasma")
        gradplotNM = bgs.plotting.GradientLinePlot(ax[1], cmap="plasma")

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            #if i>9:break
            cdc = bgs.analysis.ChildSimData.load_from_file(c)
            #N_in_30pc = [len(v)/cdc.particle_count["stars"] for v in cdc.stellar_shell_inflow_velocity.values()]
            try:
                if cdc.binary_merger_remnant["merged"]:
                    gradplotM.add_data(cdc.snapshot_times+cdc.parent_quantities["perturb_time"], cdc.stars_in_loss_cone, cdc.loss_cone)
                    nanarr = np.full_like(cdc.loss_cone, np.nan)
                    #gradplotNM.add_data(nanarr, nanarr, cdc.snapshot_times+cdc.parent_quantities["perturb_time"])
                    gradplotNM.add_data(nanarr, nanarr, cdc.loss_cone)
                else:
                    gradplotNM.add_data(cdc.snapshot_times+cdc.parent_quantities["perturb_time"], cdc.stars_in_loss_cone, cdc.loss_cone)
                    nanarr = np.full_like(cdc.loss_cone, np.nan)
                    gradplotM.add_data(nanarr, nanarr, cdc.loss_cone)
                    #gradplotM.add_data(nanarr, nanarr, cdc.snapshot_times+cdc.parent_quantities["perturb_time"])
                #gradplot.add_data(cdc.snapshot_times+cdc.parent_quantities["perturb_time"], cdc.loss_cone, cdc.stars_in_loss_cone, marker=("o" if cdc.binary_merger_remnant["merged"] else "s"))
            except AttributeError:
                print(f"\nskipping {c}")
                continue
        gradplotM.plot(logcolour=True)
        gradplotNM.plot(logcolour=True)
        gradplotM.add_cbar(label="Loss Cone J")
        gradplotNM.add_cbar(label="Loss Cone J")
        ax[0].text(0.85, 0.9, "Merged", transform=ax[0].transAxes, horizontalalignment="right")
        ax[1].text(0.85, 0.9, "Not Merged", transform=ax[1].transAxes, horizontalalignment="right")
        ax[1].set_xlabel("Age/Myr")
        ax[0].set_ylabel("Stars in Loss Cone")
        ax[1].set_ylabel("Stars in Loss Cone")
        #ax[0].set_ylabel("Diff. J of BHB and Stars")
        #ax[1].set_ylabel("Diff. J of BHB and Stars")
        #ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        plt.show()
    
    if False:
        print("Call {}".format(call))
        call += 1

        #fig, ax = plt.subplots(1,2)
        rawdat = []
        counter=0

        for i, c in enumerate(cubes):
            #print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            #if i>19:break
            cdc = bgs.analysis.ChildSimData.load_from_file(c)
            try:
                if not cdc.relaxed_remnant_flag:
                    print(f"\nWarning! \n{c} not relaxed")
                    continue
                theta_shifted = cdc.ang_mom_diff_angle-np.pi/2
                rawdat.append([
                            cdc.snapshot_times[-1]+cdc.parent_quantities["perturb_time"],
                            np.sum(np.abs(np.diff(np.sign(theta_shifted)))>1),
                            c.split("/")[-1].replace("cube-", "").replace(".hdf5", "")[:-4]
                            #cdc.binary_merger_remnant["merged"]
                ])
                #print(f"{c} -> {rawdat[counter][2]}")
                counter += 1
            except AttributeError:
                print(f"\nskipping {c}")
                continue
        
        data = pd.DataFrame(data=rawdat, columns=["Final Time", "Number of Flips", "Merged"])
        j = sns.jointplot(data=data, x="Final Time", y="Number of Flips", hue="Merged")
        j.ax_joint.set_ylim(-1, data["Number of Flips"].max()+1)

        #ax[0].scatter(endtime[merged], flips[merged], label="merged")
        #ax[0].scatter(endtime[~merged], flips[~merged], label="not merged")
        #for m in (True, False):
        #    mask = np.logical_and(m==merged, flips>-1)
        #    kde = scipy.stats.gaussian_kde(flips[mask])
        #    xpts = np.linspace(0, 20, 1000)
        #    ax[1].plot(xpts, kde.evaluate(xpts))

        #ax[1].hist(flips[merged], bins=np.arange(flips.max()+1), alpha=0.6, density=True)
        #ax[1].hist(flips[~merged], bins=np.arange(flips.max()+1), alpha=0.6, density=True)
        #ax[0].legend()
        plt.show()


    if False:
        radcut = 600
        snaplist = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/C-D-3.0-0.005/perturbations/001/output")
        escapers = np.full_like(snaplist, np.nan, dtype=float)
        for i, snapfile in enumerate(snaplist):
            if i < 50: continue
            fig, ax = plt.subplots(1,1)
            snap = pygad.Snapshot(snapfile, physical=True)
            snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
            snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")
            ballmask = pygad.BallMask(pygad.UnitScalar(radcut, "kpc"))
            vmag = pygad.utils.geo.dist(snap.stars[ballmask]["vel"])
            r = snap.stars[ballmask]["r"]

            '''vesc_fun = bgs.analysis.escape_velocity(snap[ballmask])
            vesc = vesc_fun(r)
            escapers[i] = np.sum(vmag>vesc)'''
            
            p = ax.hist2d(r, vmag, bins=(np.arange(0, radcut, 2), np.arange(0, 4000, 40)), norm=colors.LogNorm(vmin=0.1, clip=True))
            ax.set_xlabel("r/kpc")
            ax.set_ylabel("|v|/km/s")

            # escape velocity
            vesc = bgs.analysis.escape_velocity(snap)
            r_ = np.linspace(1e-1, radcut*0.99, 300)
            ax.plot(r_, vesc(r_), c="tab:red", label=r"$v_\mathrm{esc}$")
            #plt.hist(vmag, bins=np.arange(0, 3500, 100), histtype="step")
            #plt.yscale("log")
            plt.colorbar(p[3], label="Count")
            ax.text(0.02, 0.955, f"{bgs.general.convert_gadget_time(snap):.2f} Gyr", ha="left", bbox={"ec":"k", "fc":"w"}, transform=ax.transAxes)
            ax.legend()
            plt.show()
            quit()
        
        plt.plot(escapers, "-o")
        plt.show()
    
    if False:
        print("Call {}".format(call))
        call += 1

        fig, ax = plt.subplots(1,1)

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            #if i>19:break
            cdc = bgs.analysis.ChildSimData.load_from_file(c)
            if not cdc.relaxed_remnant_flag:
                    print(f"\nWarning! \n{c} not relaxed")
                    continue
            try:
                ax.plot(cdc.snapshot_times+cdc.parent_quantities["perturb_time"], cdc.num_escaping_stars)
            except:
                print(f"\nskipping {c}")
                continue
        plt.show()