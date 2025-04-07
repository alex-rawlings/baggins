import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_table("local_dwarf.csv", sep=",", header=0)
data2 = pd.read_table("NGC3115_DGs.csv", sep=",", header=0, skipinitialspace=True)
data3 = pd.read_table("mcconnachie12.csv", sep=",", header=0, skipinitialspace=True)
data4 = pd.read_table("carlsten20.csv", sep=",", header=0, skipinitialspace=True)
data5a = pd.read_fwf("GCs_dists.txt", skiprows=13)
data5b = pd.read_fwf("GCs_struct.txt", skiprows=18)
data6 = pd.read_table("price.csv", sep=",", header=0, skipinitialspace=True)
data7 = pd.read_table("misgeld_09.csv", sep=",", header=0, skipinitialspace=True)
data8 = pd.read_table("matkovic05.csv", sep=",", header=0, skipinitialspace=True)

# combine datasets
data5 = pd.merge(data5a, data5b, on="ID")

#pd.set_option('display.max_rows', None)

# these are taken from 'intruder_satellite.py'
sim_mass = 4.11e7
sim_re = 163 # pc
sim_sigma = 335.25

fig, ax = plt.subplots(1, 3)
fig.set_figwidth(2.5 * fig.get_figwidth())
ax[0].sharey(ax[1])
ax[1].sharex(ax[2])

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

def convert_modulus_to_pc(m):
    return 10**(1 + m / 5)

ax[0].scatter(sim_mass, sim_re, label="BH cluster", color="tab:red", s=120, zorder=1)
# cluster with BH, as from `intruder_satellite.py`
ax[1].scatter(sim_sigma, sim_re, color="tab:red", s=120, zorder=1)

ax[2].scatter(sim_sigma, sim_mass, color="tab:red", s=120, zorder=1)

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

ax[0].errorbar(
    data6.loc[:, "Mdyn"]*1e8,
    data6.loc[:, "Re"],
    yerr = data6.loc[:, "Re_err"],
    fmt="v", label="Price+09",
    color="gray", alpha=0.4
)

y = convert_ang_size_to_pc(convert_modulus_to_pc(33.28), data7.loc[:, "Re"])
ax[0].errorbar(
    convert_appmag_to_mass(data7.loc[:, "V0"], d=convert_modulus_to_pc(33.28)),
    convert_ang_size_to_pc(convert_modulus_to_pc(33.28), data7.loc[:, "Re"]),
    yerr = [
        y-convert_ang_size_to_pc(convert_modulus_to_pc(33.28), data7.loc[:, "Re"] - data7.loc[:, "Re_err"]),
        convert_ang_size_to_pc(convert_modulus_to_pc(33.28), data7.loc[:, "Re"] + data7.loc[:, "Re_err"]) - y
    ],
    fmt="h", label="Misgeld+09",
    color="gray", alpha=0.4
)

# add LEGUS sample
x_legus = np.geomspace(1e2, 1e5, 100)
y_legus_L = 10**(
    np.log10(2.548) + 0.242*np.log10(x_legus/1e4) - 0.25
)
y_legus_U = 10**(
    np.log10(2.548) + 0.242*np.log10(x_legus/1e4) + 0.25
)
ax[0].fill_between(x_legus, y_legus_L, y_legus_U, alpha=0.4, label="LEGUS")


ax[1].errorbar(data5.loc[:,"sig_v"], 1e3*data5.loc[:,"R_Sun"]* np.tan(data5.loc[:,"r_h"] / 60 * np.pi/ 180), xerr=data5.loc[:, "+/-.1"], ls="", fmt="o", label="GCs", color="gray", alpha=0.4)

ax[2].errorbar(
    data8.loc[:, "sigma"],
    convert_absmag_to_mass(data8.loc[:, "b_abs"]),
    xerr = data8.loc[:, "sigma_err"],
    fmt="s", label="Matkovic+05", color="gray", alpha=0.4
)

ax[0].legend(loc="lower right")
ax[1].legend()
ax[2].legend()
for axi in ax:
    axi.set_xscale("log")
    axi.set_yscale("log")
ax[0].set_xlabel("Mstar / Msun")
ax[0].set_ylabel("Reff / pc")
ax[1].set_xlabel(r"$\sigma_\star$")
ax[1].set_ylabel("Reff / pc")
ax[2].set_xlabel(r"$\sigma_\star$")
ax[2].set_ylabel("Mstar / Msun")

plt.savefig("compact.png", dpi=300)