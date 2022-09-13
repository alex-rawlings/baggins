import argparse
import os.path
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import json
import cm_functions as cmf


#set up the command line options
parser = argparse.ArgumentParser(description="Visually compare galaxy ICs", allow_abbrev=False)
parser.add_argument(type=str, help="path to comparison parameter file", dest="path")
args = parser.parse_args()

SL = cmf.CustomLogger("script_log", console_level="INFO")
marker_kwargs = {"zorder":10, "edgecolor":"k", "linewidth":0.5}

#read in literature data
#BH - Bulge relation
bh_data = cmf.literature.LiteratureTables("sahu_2020")

#BH - sigma relation
BHsigmaData = cmf.literature.LiteratureTables("vdBosch_2016")

#inner DM data
fDMData = cmf.literature.LiteratureTables("jin_2020")


cols = cmf.plotting.mplColours()
alpha = 0.4
legend_font_size = "x-small"
markersz = 1
linewd = 1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7,3))

ax1.set_xlim(8.7, 12.5)
ax1.set_ylim(7.2, 11)
ax1.set_xlabel(r"log(M$_\mathrm{bulge}$/M$_\odot$)")
ax1.set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
bh_data.scatter("logM*_sph", "logMbh", xerr="logM*_sph_ERR", yerr="logMbh_ERR", ax=ax1, mask=bh_data.table.loc[:,"Cored"], label="Cored")
bh_data.scatter("logM*_sph", "logMbh", xerr="logM*_sph_ERR", yerr="logMbh_ERR", ax=ax1, mask=~bh_data.table.loc[:,"Cored"], scatter_kwargs={"marker":"s"}, label=r"S$\acute\mathrm{e}$rsic")

logmstar_seq = np.linspace(8, 12, 500)
ax1.plot(logmstar_seq, cmf.literature.Sahu19(logmstar_seq), c="k", alpha=0.4)
#ax1.set_title("Bulge - BH Mass", fontsize="small")

BHsigmaData.scatter("logsigma", "logBHMass", xerr="e_logsigma", yerr=("e_logBHMass", "E_logBHMass"), ax=ax2)
#ax2.legend(fontsize=legend_font_size, loc="upper left")
ylims = list(ax2.get_ylim())
ylims[0] = 8
ax2.set_ylim(ylims)
xlims = list(ax2.get_xlim())
xlims[0] = 2
ax2.set_xlim(xlims)
ax2.set_xlabel(r"log($\sigma_*$/ km/s)")
ax2.set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
#ax2.set_title("BH Mass - Stellar Dispersion", fontsize="small")

binned_fdm = scipy.stats.binned_statistic(fDMData.table.loc[:, "log(M*/Msun)"], values=fDMData.table.loc[:,"f_DM"], bins=5, statistic="median")
fDMData.scatter("log(M*/Msun)", "f_DM", ax=ax3)
fdm_radii = cmf.mathematics.get_histogram_bin_centres(binned_fdm[1])
ax3.plot(fdm_radii, binned_fdm[0], "-x", c="k", label="Median")
ax3.set_xlabel(r"log(M$_*$/M$_\odot$)")
ax3.set_ylabel(r"f$_\mathrm{DM}(r<1\,$R$_\mathrm{e})$")
#ax3.set_title("Inner DM Fraction", fontsize="small")
ax3.legend(fontsize="x-small", loc="upper left")

with open(args.path, "r") as f:
    gal_dict = json.load(f)

for lab, gal_params in gal_dict["galaxies"].items():
    SL.logger.info(f"Reading from {gal_params}")
    galaxy = cmf.initialise.GalaxyIC(parameter_file=gal_params)
    ax1.scatter(galaxy.stars.log_total_mass, galaxy.bh.log_mass, marker=("o" if galaxy.parameters.stellarCored else "s"), label=lab, **marker_kwargs)
    ax2.scatter(np.log10(galaxy.parameters.LOS_vel_dispersion), galaxy.bh.log_mass, **marker_kwargs)
    ax3.scatter(galaxy.stars.log_total_mass, galaxy.parameters.inner_DM_fraction, **marker_kwargs)
ax1.legend(fontsize="x-small", loc="upper left")
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "compic.png"), fig=fig)
plt.show()