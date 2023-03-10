import numpy as np
import matplotlib.pyplot as plt
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

analysis_params = cmf.utils.read_parameters(apf)

ecc = []
Nhalf = []
counts = np.zeros(len(datapaths))


for i, d in enumerate(datapaths):
    HMQ_files = cmf.utils.get_files_in_dir(d)
    k = cmf.analysis.KeplerModelSimple(None, None, "", None)
    k.extract_data(HMQ_files, analysis_params)
    counts[i] = k.num_groups
    ecc.append(k.obs_collapsed["e"])
    Nhalf.append([])
    for f in HMQ_files:
        hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
        Nhalf[i].append(np.nanmedian(hmq.masses_in_galaxy_radius["stars"])/hmq.particle_masses["stars"])

print(f"counts: {counts}")

ecc_std = []
Nhalf_mean = []
for e, n in zip(ecc, Nhalf):
    #ecc_std.append(np.nanstd(np.log10(e)))
    ecc_std.append(cmf.mathematics.iqr(np.log10(e)))
    Nhalf_mean.append(np.nanmedian(n))

cmapper, sm = cmf.plotting.create_normed_colours(0.5, 10.5, cmap="tab10_r")

fig, ax = plt.subplots(1,1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$N_\star$")
ax.set_ylabel(r"$\mathrm{IQR}_{\log_{10}(\mathrm{ecc})}$")
ax.scatter(Nhalf_mean, ecc_std, c=cmapper(counts), s=160, lw=0.5, ec="k")
cbar = plt.colorbar(sm, ax=ax, label="# Runs")

plt.show()