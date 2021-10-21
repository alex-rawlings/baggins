import argparse
import os.path
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import cm_functions as cmf


#set up the command line options
parser = argparse.ArgumentParser(description="Visually compare galaxy ICs", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="path")
parser.add_argument("-g", "--gal", type=str, help="name of IC", dest="gals", action="append")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose printing in script")
args = parser.parse_args()


#read in literature data
#BH - Bulge relation
mass_data = pd.read_table("./literature_data/sdss_z.csv", sep=",")
bh_data = pd.read_table("./literature_data/sahu_20.txt", sep=",", header=0)
#restrict to only ETGs (exclude also S0)
bh_data = bh_data.loc[np.logical_or(bh_data.loc[:,"Type"]=="E", bh_data.loc[:,"Type"]=="ES"), :]
cored_galaxies = np.zeros(bh_data.shape[0], dtype="bool")
for ind, gal in enumerate(bh_data.loc[:,"Galaxy"]):
    if gal[-1] == "a":
        cored_galaxies[ind] = 1
bh_data.insert(2, "Cored", cored_galaxies)
cmf.utils.create_error_col(bh_data, "logM*_sph")
cmf.utils.create_error_col(bh_data, "logMbh")

#BH - sigma relation
BHsigmaData = pd.read_table("./literature_data/bosch_16.txt", sep=";", header=0, skiprows=[1])
BHsigmaData.loc[BHsigmaData.loc[:,"e_logBHMass"]==BHsigmaData.loc[:,"logBHMass"], "e_logBHMass"] = np.nan
BHsigmaData.loc[BHsigmaData.loc[:,"logBHMass"]<1, "logBHMass"] = np.nan

#inner DM data
fDMData = pd.read_fwf("./literature_data/jin_2020.dat", comment="#", names=["MaNGAID", "log(M*/Msun)", "Re(kpc)", "f_DM", "p_e", "q_e", "T_e", "f_cold", "f_warm", "f_hot", "f_CR", "f_prolong", "f_CRlong", "f_box", "f_SR"])


cols = cmf.plotting.mplColours()
alpha = 0.6
legend_font_size = "x-small"
markersz = 1
linewd = 1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7,3))

ax1.set_xlim(8.7, 12.5)
ax1.set_ylim(7.2, 11)
ax1.set_xlabel(r"log(M$_\mathrm{bulge}$/M$_\odot$)")
ax1.set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
ax1.errorbar(bh_data.loc[bh_data.loc[:,"Cored"], "logM*_sph"], bh_data.loc[bh_data.loc[:,"Cored"], "logMbh"], xerr=bh_data.loc[bh_data.loc[:,"Cored"],"logM*_sph_ERR"], yerr=bh_data.loc[bh_data.loc[:,"Cored"],"logMbh_ERR"], ls="", marker=".", elinewidth=linewd/1.2, alpha=alpha,  label="Cored", zorder=3, c="k")
ax1.errorbar(bh_data.loc[~bh_data.loc[:,"Cored"], "logM*_sph"], bh_data.loc[~bh_data.loc[:,"Cored"], "logMbh"], xerr=bh_data.loc[~bh_data.loc[:,"Cored"],"logM*_sph_ERR"], yerr=bh_data.loc[~bh_data.loc[:,"Cored"],"logMbh_ERR"], ls="", marker="s", elinewidth=linewd/1.2, alpha=alpha,  label=r"S$\acute\mathrm{e}$rsic", zorder=3, c="k")
logmstar_seq = np.linspace(8, 12, 500)
ax1.plot(logmstar_seq, cmf.literature.Sahu19(logmstar_seq), c="k", alpha=0.4)
ax1.set_title("Bulge - BH Mass", fontsize="small")

ax2.errorbar(BHsigmaData.loc[:,"logsigma"], BHsigmaData.loc[:,"logBHMass"], xerr=BHsigmaData.loc[:,"e_logsigma"], yerr=[BHsigmaData.loc[:,"e_logBHMass"], BHsigmaData.loc[:,"E_logBHMass"]], marker=".", ls="None", elinewidth=0.5, capsize=0, color="k", alpha=alpha, ms=markersz, zorder=1, label="Bosch+16")
ax2.legend(fontsize=legend_font_size)
ax2.set_xlabel(r"log($\sigma_*$/ km/s)")
ax2.set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
ax2.set_title("BH Mass - Stellar Dispersion", fontsize="small")

binned_fdm = scipy.stats.binned_statistic(fDMData.loc[:, "log(M*/Msun)"], values=fDMData.loc[:,"f_DM"], bins=5, statistic="median")
ax3.scatter(fDMData.loc[:, "log(M*/Msun)"], fDMData.loc[:,"f_DM"], c="k", alpha=alpha, s=3, label="Jin+20")
fdm_radii = cmf.mathematics.get_histogram_bin_centres(binned_fdm[1])
ax3.plot(fdm_radii, binned_fdm[0], "-x", c="k", label="Median")
ax3.set_xlabel(r"log(M$_*$/M$_\odot$)")
ax3.set_ylabel(r"f$_\mathrm{DM}(r<1\,$R$_\mathrm{e})$")
ax3.set_title("Inner DM Fraction", fontsize="small")
ax3.legend(fontsize="x-small", loc="upper left")

for gal in args.gals:
    pfv = cmf.utils.read_parameters("{}/{}.py".format(args.path, gal))
    galaxy = cmf.initialise.galaxy_ic_base(pfv, stars=True, dm=True, bh=True)
    galaxy.dm.peak_mass = (args.verbose, None)
    galaxy.bh.mass = (args.verbose, None)
    ax1.scatter(galaxy.stars.log_total_mass, galaxy.bh.log_mass, zorder=10, label=gal, marker=("o" if pfv.stellarCored else "s"))
    ax2.scatter(np.log10(pfv.LOS_vel_dispersion), np.log10(pfv.BH_mass), zorder=10)
    ax3.scatter(galaxy.stars.log_total_mass, pfv.inner_DM_fraction, zorder=10)
ax1.legend(fontsize="x-small", loc="upper left")
figname = os.path.join(cmf.FIGDIR, "compic.png")
plt.savefig(figname)
plt.show()