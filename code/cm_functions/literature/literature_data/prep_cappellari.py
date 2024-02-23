import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ketju_functions as kf

cappellari_13_a = pd.read_fwf("cappellari_13_a.txt", header=0, na_values=("--", "----"))
cappellari_13_b = pd.read_fwf("cappellari_13_b.txt", header=0, na_values=("--", "----"))
cappellari = cappellari_13_a.merge(cappellari_13_b, on="Galaxy", how="inner")
cappellari.dropna(inplace=True)

# print(cappellari.columns)

z = 0
mstar = (
    10
    ** (
        cappellari.loc[:, "logML_star"].astype(float)
        + cappellari.loc[:, "logLum"].astype(float)
    )
).to_numpy()
# print(mstar)

peak_mass = np.full_like(mstar, np.nan)
for ind, m in enumerate(mstar):
    peak_mass[ind] = 10 ** kf.Behroozi19(m, [1e8, 1e15], z=0, plotting=False)
print(peak_mass)
