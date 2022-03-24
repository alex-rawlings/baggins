import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import cm_functions as cmf
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
        cdc = cmf.analysis.ChildSimData.load_from_file(c)

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
        for zi in np.unique(df.loc[:,zlabel]):
            mask = df.loc[:,zlabel] == zi
            p = plt.scatter(df.loc[mask,xlabel], df.loc[mask,ylabel], label=zi, linewidths=sclw, edgecolors="k")
        plt.legend(title=zlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if earlyreturn:
            return df
    plt.savefig(os.path.join(cmf.FIGDIR, "analysis-explore/{}-{}.png".format(xlabel, ylabel)))
    return p, df
    #plt.show()
    #quit()


if __name__ == "__main__":
    cubes = cmf.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes", recursive=True)
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
        jp.ax_joint.plot(rvir, cmf.literature.Kratsov13(rvir))
        plt.show()
    
    if True:
       # misgeld = pd.read_table("../../initialise_scripts/literature_data/misgeld_11.dat", sep="|", header=0, skiprows=[1,3], skipinitialspace=True, comment="#")
        print("Call {}".format(call))
        call += 1
        attrs = ["virial_info", "total_stellar_mass", "merger_name"]
        df = extract_and_plot_point(attrs, xdkey="mass", categorical_y=True)
        plt.xlim(1e13, 2e14)
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(os.path.join(cmf.FIGDIR, "analysis-explore/{}-{}.png".format(attrs[0]+"_mass", attrs[1])))
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["relaxed_effective_radius", "relaxed_inner_DM_fraction", "merger_name"]
        extract_and_plot_point(attrs)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_merger_timescale", "merger_name", "binary_merger_remnant"]
        extract_and_plot_point(attrs, categorical_y=True)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_merger_timescale", "binary_spin_flip", "binary_merger_remnant"]
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
        cols = cmf.plotting.mplColours()

        fig, ax = plt.subplots(1,1)
        ax.axhline(np.pi/2, c="k", ls="--")
        ax.set_ylim(0, np.pi)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\theta$")

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            cdc = cmf.analysis.ChildSimData.load_from_file(c)
            #marker = "o" if cdc.binary_merger_remnant["merged"] else ""
            alpha = 0.9  if cdc.binary_merger_remnant["merged"] else 0.3
            ax.plot(cdc.snapshot_times, cdc.ang_mom_diff_angle, c=cols[int(cdc.binary_spin_flip)], alpha=alpha)
        plt.show()

    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_merger_timescale", "parent_quantities", "binary_spin_flip"]
        extract_and_plot_point(attrs, ydkey="initial_e", zdkey=None)

    if False:
        print("Call {}".format(call))
        call += 1

        fig, ax = plt.subplots(1,1)
        #colour lines by their age
        cmapper, sm = cmf.plotting.create_normed_colours(0, 13.8e3)

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            cdc = cmf.analysis.ChildSimData.load_from_file(c)
            age = cdc.parent_quantities["perturb_time"] + cdc.binary_merger_timescale
            ax.loglog(cdc.radial_bin_centres["stars"], cdc.relaxed_density_profile["stars"], c=cmapper(age))
        plt.colorbar(sm)
        plt.show()