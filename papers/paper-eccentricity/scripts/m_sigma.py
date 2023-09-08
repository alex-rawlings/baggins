import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


gal_dir = "/scratch/pjohanss/arawling/collisionless_merger/galaxies/dehnen"

gals = cmf.utils.get_files_in_dir(gal_dir, recursive=True)

fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$\log_{10}(\sigma/[\mathrm{km/s}])$")
ax.set_ylabel(r"$\log_{10}(M_\bullet/\mathrm{M}_\odot)$")
cols = cmf.plotting.mplColours()

# add data from observations
BHsigmaData = cmf.literature.LiteratureTables("vdBosch_2016")
ax.errorbar(BHsigmaData.table.loc[:,"logsigma"], BHsigmaData.table.loc[:,"logBHMass"], xerr=BHsigmaData.table.loc[:,"e_logsigma"], yerr=[BHsigmaData.table.loc[:,"e_logBHMass"], BHsigmaData.table.loc[:,"E_logBHMass"]], marker=".", ls="None", elinewidth=0.5, capsize=0, zorder=1, label="Bosch+16", c="k", alpha=0.4)


for g in gals:
    if "_a" not in g or "1M" not in g: continue
    print(f"Reading: {g}")
    snap = pygad.Snapshot(g, physical=True)
    sigma = 0
    mask = pygad.BallMask(7)
    for i in range(3):
        sigma += pygad.analysis.los_velocity_dispersion(snap[mask], proj=i)**2
    sigma = np.sqrt(np.nanmedian(sigma))
    if "0001" in g:
        marker = "s"
        label = r"$M_\bullet=1\times10^7$"
    elif "0005" in g:
        marker = "d"
        label = r"$M_\bullet=5\times10^7$"
    else:
        marker = "o"
        label = r"$M_\bullet=1\times10^8$"

    ax.scatter(np.log10(sigma), np.log10(snap.bh["mass"]), marker=marker, s=80, label=label, zorder=10)
ax.legend()
plt.savefig("m-sig.pdf")
plt.show()