import numpy as np
import h5py
from cm_functions.mathematics import radial_separation


rng = np.random.default_rng(42)

near_star = False

if near_star:
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/high-time-output/star_perturb/near_star/ic.hdf5"
    r_boundary = 0.05
else:
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/high-time-output/star_perturb/distant_star/ic.hdf5"
    r_boundary = 1.0

with h5py.File(snapfile, "r+") as f:
    pos = f["/PartType4/Coordinates"][:]
    bh_pos = f["/PartType5/Coordinates"][0,:]
    r = radial_separation(pos, bh_pos)
    mask = r < r_boundary
    print(f"Inner stars: {r[mask].shape[0]}")
    print(f"Outer stars: {r[~mask].shape[0]}")
    selected_star_idx = rng.choice(pos[mask].shape[0])
    print("Before perturbing: ")
    print(f["/PartType4/Coordinates"][selected_star_idx,:])
    #f["/PartType4/Coordinates"][selected_star_idx, 0] += 10

with h5py.File(snapfile, "r") as f:
    print("After perturbing: ")
    print(f["/PartType4/Coordinates"][selected_star_idx,:])