import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cm_functions as cmf



data = pd.read_table("/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/runinfo.txt", sep=",", header=0, parse_dates=[1])
data.drop("Unnamed: 4", axis="columns", inplace=True)
etas = [0.002, 0.002, 0.002, 0.010, 0.020, 0.030, 0.050]
epsilons = [0.0035, 0.0200, 0.0300, 0.0035, 0.0035, 0.0035, 0.0035]
data["etas"] = etas
data["epsilons"] = epsilons

cmf.utils.add_time_column(data, unit="h", colname="Elapsed", newcolname="Elapsed_s")

print(data)

#markers = np.full_like(data["etas"].shape[0], "", dtype=str)
sizes = [200 if v == "COMPLETED" else 50 for v in data["State"]]

s = plt.scatter(data.loc[:, "etas"], data.loc[:, "epsilons"], c=data.loc[:, "Elapsed_s"], s=sizes)

cbar = plt.colorbar(s)
cbar.ax.set_ylabel("Walltime [h]")
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\varepsilon_\star$")
plt.title("Walltime to Reach 1 Gyr")
plt.savefig("{}/errtolintacc/time1gyr.png".format(cmf.FIGDIR))
#plt.show()