import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cm_functions as cmf


if __name__ == "__main__":
    cubes = cmf.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes", recursive=True)
    num_sims = len(cubes)
    rawdat = []

    for i, c in enumerate(cubes):
        print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
        cdc = cmf.analysis.ChildSimData.load_from_file(c)
        if i==100:
            print(cdc.bh_masses)
            print(cdc.relaxed_half_mass_radius)
            print(cdc.relaxed_effective_radius)
            print(cdc.relaxed_stellar_velocity_dispersion)
            plt.plot(cmf.mathematics.get_histogram_bin_centres(np.geomspace(2e-1,20,51)), cdc.relaxed_density_profile)
            plt.show()
            quit()
        rawdat.append([
                        c.split("/")[-1],
                        cdc.relaxed_effective_radius["estimate"],
                        cdc.binary_merger_timescale,
                        (cdc.binary_merger_remnant["merged"])
        ])
    print("\nComplete")

    df = pd.DataFrame(rawdat, columns=["names", "x", "y", "merged"])
    pd.set_option("display.max_rows", 500)
    #print(df)
    sns.jointplot(data=df, x="x", y="y", hue="merged")
    plt.show()

    """
    cols = cmf.plotting.mplColours()
    for i in np.unique(z):
        mask = z==i
        plt.scatter(x[mask], y[mask], c=cols[i], edgecolors="k")
    plt.show()"""