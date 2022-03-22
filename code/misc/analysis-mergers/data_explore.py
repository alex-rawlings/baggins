import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cm_functions as cmf
import pygad


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

def extract_and_plot_point(attrs, xdkey="estimate", ydkey="estimate", zdkey="merged", categorical_y=False):
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
            plt.scatter(df.loc[mask,xlabel], df.loc[mask,ylabel], label=zi)
        plt.legend(title=zlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.savefig(os.path.join(cmf.FIGDIR, "analysis-explore/{}-{}.png".format(xlabel, ylabel)))
    return p
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
        extract_and_plot_point(attrs, ydkey="radius")
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["virial_info", "relaxed_half_mass_radius", "merger_name"]
        jp = extract_and_plot_point(attrs, xdkey="radius")
        rvir = np.linspace(300, 800, 1000)
        jp.ax_joint.plot(rvir, cmf.literature.Kratsov13(rvir))
        plt.show()

    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["relaxed_effective_radius", "relaxed_inner_DM_fraction", "binary_merger_remnant"]
        extract_and_plot_point(attrs)
    
    if False:
        print("Call {}".format(call))
        call += 1
        attrs = ["binary_merger_timescale", "merger_name", "binary_merger_remnant"]
        extract_and_plot_point(attrs, categorical_y=True)

    if False:
        print("Call {}".format(call))
        call += 1

        fig, ax = plt.subplots(1,1)
        r = cmf.mathematics.get_histogram_bin_centres(np.geomspace(2e-2,2e1,51))

        for i, c in enumerate(cubes):
            print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
            cdc = cmf.analysis.ChildSimData.load_from_file(c)

            ax.loglog(r, cdc.relaxed_density_profile)
        plt.show()