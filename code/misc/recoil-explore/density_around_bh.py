import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


snapfiles = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")

snap_idx = bgs.general.snap_num_for_time(snapfiles, 0.32, "Gyr")

print(f"We are going to use snapshot {snap_idx} of {len(snapfiles)}")

snap = pygad.Snapshot(snapfiles[snap_idx], physical=True)
pygad.Translation(-snap.bh["pos"][0]).apply(snap, total=True)

# define the influence radius
rinfl = list(bgs.analysis.influence_radius(snap).values())[0]

print(f"Influence radius is {rinfl:.2e}")

Sigma_rinfl = pygad.analysis.profile_dens(snap.stars, qty="mass", r_edges=[0, rinfl], proj=1)
Sigma_3rinfl = pygad.analysis.profile_dens(snap.stars, qty="mass", r_edges=[0, 3*rinfl], proj=1)

print(f"Projected density within influence radius is {Sigma_rinfl.view(np.ndarray)[0]:.2e}{Sigma_rinfl.units}, which is {Sigma_rinfl.view(np.ndarray)[0] / Sigma_3rinfl.view(np.ndarray)[0]:.2e} more dense than 3x influence radius")