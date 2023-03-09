import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf


datapaths = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H-H-3.0-0.001",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0500-H0500-3.0-0.001",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0250-H0250-3.0-0.001",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0100-H0100-3.0-0.001",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0050-H0050-3.0-0.001"
]

apf = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"

analysis_params = cmf.utils.read_parameters(apf)

ecc = []
Nhalf = []

star_masses = [1e5, 2e5, 4e5, 1e6, 2e6]

for i, d in enumerate(datapaths):
    HMQ_files = cmf.utils.get_files_in_dir(d)
    k = cmf.analysis.KeplerModelSimple(None, None, "", None)
    k.extract_data(HMQ_files, analysis_params)
    ecc.append(k.obs_collapsed["e"])
    Nhalf.append([])
    for f in HMQ_files:
        hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
        Nhalf[i].append(np.nanmedian(hmq.masses_in_galaxy_radius["stars"])/star_masses[i])

ecc_std = []
Nhalf_mean = []
for e, n in zip(ecc, Nhalf):
    #ecc_std.append(np.nanstd(e))
    ecc_std.append(cmf.mathematics.iqr(e))
    Nhalf_mean.append(np.nanmedian(n))


plt.scatter(Nhalf_mean, ecc_std)

plt.show()