import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dat1 = pd.read_fwf("GCs_dists.txt", skiprows=13)
dat2 = pd.read_fwf("GCs_struct.txt", skiprows=18)

# combine datasets
dat = pd.merge(dat1, dat2, on="ID")
print(dat.columns)

plt.errorbar(dat.loc[:,"sig_v"], dat.loc[:,"R_Sun"]* np.tan(dat.loc[:,"r_h"] / 60 * np.pi/ 180), xerr=dat.loc[:, "+/-.1"], ls="", fmt="o", label="GCs")

# cluster with BH, as from `intruder_satellite.py`
plt.loglog(335.25, 0.15, marker="o", label="sim.", ls="")
plt.xlabel(r"$\sigma$ [km/s]")
plt.ylabel("Half mass radius [kpc]")
plt.legend()
plt.savefig("GCs.png", dpi=300)