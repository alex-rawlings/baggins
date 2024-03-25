import numpy as np
import baggins as bgs
import matplotlib.pyplot as plt
import seaborn as sns


hmq_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_100K-D_100K-3.720-0.279"

hmq_files = bgs.utils.get_files_in_dir(hmq_dir)
merger_times = []
for f in hmq_files:
    hmq = bgs.analysis.HMQuantitiesBinaryData.load_from_file(f)
    if hmq.merger_remnant["merged"]:
        status, idx0 = hmq.idx_finder(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
        if not status: continue
        merger_times.append(hmq.binary_time[-1]-hmq.binary_time[idx0])

print(f"Mean: {np.nanmean(merger_times)}")
print(f"SD: {np.nanstd(merger_times)}")
print(f"Median: {np.nanmedian(merger_times)}")
print(f"IQR: {bgs.mathematics.iqr(merger_times)}")

sns.violinplot(merger_times)
sns.rugplot(merger_times)
plt.show()