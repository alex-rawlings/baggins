import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad




snapdir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/trail_blazer/H_2M_a-H_2M_b-30.000-2.000/output"

snapfile = bgs.utils.get_snapshots_in_dir(snapdir)[-30]

snap = pygad.Snapshot(snapfile, physical=True)

fof_group_IDs, N_groups = pygad.analysis.find_FoF_groups(snap.stars, l="20 pc", sort=True, verbose=10)


print(f"There are {N_groups} FoF groups found")
print(f"Unique FoF IDs are {np.unique(fof_group_IDs)}")

assert N_groups > 0

for i in range(N_groups):
    mask = pygad.IDMask(snap.stars[i==fof_group_IDs]["ID"])
    assert np.sum(i==fof_group_IDs) > 0
    plt.scatter(
        snap.stars[mask]["pos"][:,0],
        snap.stars[mask]["pos"][:,2],
        marker=".",
        label=i
        )
    if i>10:
        break
plt.legend(title="FoF Group ID")
plt.show()