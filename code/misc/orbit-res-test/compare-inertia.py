from baggins.utils.data_handling import save_data
import numpy as np
import matplotlib.pyplot as plt
import os
import baggins as bgs

inertia_data = "./inertia-data"
orbits = ["0-001", "0-005", "0-030", "0-180", "1-000"]
resolutions = ["fiducial", "x02", "x05", "x10"]

fig, ax = plt.subplots(5, 2, figsize=(6, 6), sharex='all', sharey='all')
cols_list = bgs.plotting.mplColours()
cols = dict(zip(resolutions, cols_list))

files = []
with os.scandir(inertia_data) as s:
    for entry in s:
        files.append(entry.path)
files.sort()

for ind, this_file in enumerate(files):
    for ind2, orbit in enumerate(orbits):
        if orbit in this_file:
            #print("Reading: {}".format(this_file))
            save_dict = bgs.utils.load_data(this_file)
            res = this_file.split("/")[-1].split("-")[1]
            init_idx = len(list(save_dict["ratios"].values())[0])+10
            below09_idx = [init_idx, init_idx]
            below09_idx_temp = [99, 99]
            for ind3, key in enumerate(save_dict["ratios"].keys()):
                ax[ind2][0].plot(
                    save_dict["time_of_snap"], save_dict["ratios"][key][:,0],
                    markevery=[-1], marker=('o' if ind3==0 else 's'), 
                    c=cols[res]
                    )
                ax[ind2][1].plot(
                    save_dict["time_of_snap"], save_dict["ratios"][key][:,1],
                    markevery=[-1], marker=('o' if ind3==0 else 's'),
                    c=cols[res], label=(res if ind2==0 and ind3==0 else "")
                    )
                for ind4 in range(2):
                    below09_idx_temp[ind4] = np.argmax(save_dict["ratios"][key][:,ind4]<0.9)
                    if below09_idx_temp[ind4] < below09_idx[ind4]:
                        below09_idx[ind4] = below09_idx_temp[ind4]
            for i in range(2):
                if min(below09_idx) > 0:
                    ax[ind2][i].axvline(save_dict["time_of_snap"][min(below09_idx)], 0, 0.1, c=cols[res])
            
ax[-1][0].set_xlabel("Time/Gyr")
ax[-1][1].set_xlabel("Time/Gyr")
ax[0][0].set_title("b/a")
ax[0][1].set_title("c/a")
ax[0][1].legend(fontsize='x-small')
for i in range(ax.shape[0]):
    ax[i][0].set_ylabel('Axis Ratio')
    ax_right = ax[i][1].twinx()
    ax_right.set_yticks([])
    ax_right.set_ylabel(orbits[i])
ax = np.concatenate(ax).flat
for axi in ax:
    axi.axhline(0.9, c='k', alpha=0.6)
plt.tight_layout()
plt.savefig("/users/arawling/figures/res-test/inertia-compare.png", dpi=300)
plt.show()