import os.path
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cm_functions as cmf


datapaths = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_1-000",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-500",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-250",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-100",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-050",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-025",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-010",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/MC_sample/H_0-005"
]

apf = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"

if True:
    cmf.plotting.set_publishing_style()
    full_figsize = rcParams["figure.figsize"]
    full_figsize[0] *= 2
else:
    full_figsize = None

analysis_params = cmf.utils.read_parameters(apf)

ecc = []
Nhalf = []
counts = np.zeros(len(datapaths))


for i, d in enumerate(datapaths):
    HMQ_files = cmf.utils.get_files_in_dir(d)
    k = cmf.analysis.KeplerModelSimple(None, None, "", None)
    k.extract_data(HMQ_files, analysis_params)
    counts[i] = k.num_groups
    try:
        k.transform_obs("e", "var_e", lambda x: [np.nanvar(x)])
    except AssertionError:
        print("------------\nNaNs detected, but will ignore for now!\n------------")
    ecc.append([v for v in k.obs["var_e"]])
    Nhalf.append(k.obs["var_e"])
    for f in HMQ_files:
        hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
        Nhalf[i].append(np.nanmedian(hmq.masses_in_galaxy_radius["stars"])/hmq.particle_masses["stars"])

print(f"counts: {counts}")

ecc_std_mean = []
ecc_std_std = []
Nhalf_mean = []
for e, n in zip(ecc, Nhalf):
    #ecc_std_mean.append(np.nanstd(np.log10(e)))
    #ecc_std_mean.append(cmf.mathematics.iqr(np.log10(e)))
    ecc_std_mean.append(np.sqrt(np.nanmean(e)))
    ecc_std_std.append(np.nanstd(e))
    Nhalf_mean.append(np.nanmedian(n))

# fit the functional form
func = lambda x, k: k/np.sqrt(x+k**2)
popt, pcov = scipy.optimize.curve_fit(func, Nhalf_mean, ecc_std_mean, sigma=ecc_std_std)
print(f"Optimal parameters: {popt}")


cmapper, sm = cmf.plotting.create_normed_colours(0.5, 10.5, cmap="tab10_r")

fig, ax = plt.subplots(1,1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$N_\star$")
ax.set_ylabel(r"$\bar{\sigma}_\mathrm{ecc}$")
#ax.scatter(Nhalf_mean, ecc_std_mean, c=cmapper(counts), s=160, lw=0.5, ec="k", zorder=10, label="Sims.")
for i, (n, e, ee, c) in enumerate(zip(Nhalf_mean, ecc_std_mean, ecc_std_std, counts)):
    ax.errorbar(n, e, yerr=ee, color=cmapper(c), label=("Sims." if i==0 else ""), fmt="o", mec="k", ls=None, lw=0.5, capsize=15, ms=10, zorder=10)
cbar = plt.colorbar(sm, ax=ax, label="# Runs")

# overlay scaling
xseq = np.geomspace(0.9*np.min(Nhalf_mean), 1.1*np.max(Nhalf_mean), 500)
ax.plot(xseq, func(xseq, popt), c="k", ls="--", lw=3, label=f"k={popt[0]:.2f}")

# set nicer y limits
cmf.plotting.nice_log10_scale(ax)
ax.legend(handletextpad=2)

plt.savefig(os.path.join(cmf.FIGDIR, "nasim_plot.pdf"))
plt.show()