import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cm_functions as cmf

if __name__ == "__main__":
    #sns.set_theme(style="dark")
    cubes = cmf.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes", recursive=True)
    num_sims = len(cubes)
    rawdat = []

    for i, c in enumerate(cubes):
        print("Reading cubes: {:.1f}%                              ".format(i/(num_sims-1)*100), end="\r")
        cdc = cmf.analysis.ChildSimData.load_from_file(c)
        rawdat.append([
                        c.split("/")[-1],
                        cdc.relaxed_effective_radius["estimate"],
                        cdc.r_hard,
                        cdc.binary_merger_remnant["merged"],
                        cdc.relaxed_remnant_flag

        ])
    print("\nComplete")

    df = pd.DataFrame(rawdat, columns=["names", "x", "y", "merged", "relaxed"])
    pd.set_option("display.max_rows", 500)
    print(df)
    sns.jointplot(data=df, x="x", y="y", hue="merged")
    plt.show()

