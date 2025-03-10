import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_table("local_dwarf.csv", sep=",", header=0)
data2 = pd.read_table("NGC3115_DGs.csv", sep=",", header=0, skipinitialspace=True)
data3 = pd.read_table("mcconnachie12.csv", sep=",", header=0, skipinitialspace=True)
data4 = pd.read_table("carlsten20.csv", sep=",", header=0, skipinitialspace=True)

#pd.set_option('display.max_rows', None)

# these are taken from 'intruder_satellite.py'
sim_mass = 4.11e7
sim_re = 163 # pc

fig, ax = plt.subplots(1, 2)

def convert_absmag_to_mass(mag, ML=4):
    # convert mag to lum
    Mag_sol = 4.83
    lum = 10**((mag - Mag_sol) / -2.5)
    return lum / ML

def convert_appmag_to_mass(mag, d, ML=4):
    # need abs. mag
    M = mag - 5 * (np.log10(d) - 1)
    return convert_absmag_to_mass(M, ML)

def convert_ang_size_to_pc(d, t):
    return np.tan(t * np.pi / (3600 * 180)) * d

ax[0].scatter(sim_mass, sim_re, label="BH cluster", color="tab:red", s=120, zorder=1)

ax[0].scatter(data1.loc[:, "M_star (Msun)"], data1.loc[:, "R_eff (pc)"], label="Misgeld+11", c="gray", alpha=0.4)

ax[0].errorbar(
    convert_absmag_to_mass(data2.loc[:, "Mag"]),
    data2.loc[:, "Re"],
    xerr=convert_absmag_to_mass(data2.loc[:, "Mag"]+data2.loc[:, "Mag_err"]),
    yerr=data2.loc[:, "Re_err"],
    fmt="s", label="Canossa-Gosteinski+24",
    color="gray", alpha=0.4
)

ax[0].errorbar(
    convert_appmag_to_mass(data4.loc[:, "magG"], data4.loc[:, "dist"]*1e6),
    convert_ang_size_to_pc(data4.loc[:, "dist"]*1e6, data4.loc[:, "re"]),
    xerr=convert_appmag_to_mass(data4.loc[:, "magG"]+data4.loc[:, "magG_err"], data4.loc[:, "dist"]*1e6),
    yerr=convert_ang_size_to_pc(data4.loc[:, "dist"]*1e6, data4.loc[:, "re_err"]),
    fmt="^", color="gray", alpha=0.4,
    label="Carlsten+20"
)

ax[0].legend()
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Mstar / Msun")
ax[0].set_ylabel("Reff / pc")

plt.show()