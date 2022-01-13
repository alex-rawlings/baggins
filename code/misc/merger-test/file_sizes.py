import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dat = pd.read_table("/scratch/pjohanss/arawling/collisionless_merger/mergers/file_info.txt", sep="\t", header=None, names=["size", "file"])
sizes_GB = np.full_like(dat.loc[:,"size"], np.nan, dtype=float)

for p, m in zip(("K", "M", "G", "T"), (1e-6, 1e-3, 1, 1e3)):
    mask = np.array([True if d[-1]==p else False for d in dat.loc[:,"size"]])
    s = np.array([d[:-1] for d in dat.loc[mask, "size"]], dtype=float)
    sizes_GB[mask] = s * m

fig, ax = plt.subplots(1,1,figsize=(8, 3))
ax.plot(dat.loc[:,"file"][:-1], sizes_GB[:-1])
plt.xticks(rotation=90, fontsize="xx-small")
plt.show()