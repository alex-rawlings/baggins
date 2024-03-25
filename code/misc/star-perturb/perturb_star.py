import numpy as np
import h5py
from baggins.mathematics import radial_separation


rng = np.random.default_rng(42)

near_particle = True
perturbation = 0.01 # kpc
parttype = "star"

particles = {"star":"PartType4", "dm":"PartType1"}

if near_particle:
    snapfile = f"/scratch/pjohanss/arawling/gadget4-ketju/A-C-3.0-0.05/perturbations/near_{parttype}_{perturbation}/snap_050.hdf5"
    r_boundary = [0, 0.05]
    print(f"Perturbing a particle between {r_boundary[0]} and {r_boundary[1]} kpc...")
else:
    snapfile = f"/scratch/pjohanss/arawling/gadget4-ketju/A-C-3.0-0.05/perturbations/distant_{parttype}_{perturbation}/snap_050.hdf5"
    r_boundary = [0.1, 10.0]
    print(f"Perturbing a {particles[parttype]} particle between {r_boundary[0]} and {r_boundary[1]} kpc...")
print(f"IC file: {snapfile}")

with h5py.File(snapfile, "r+") as f:
    pos = f[f"/{particles[parttype]}/Coordinates"][:]
    bh_pos_1 = f["/PartType5/Coordinates"][0,:]
    bh_pos_2 = f["/PartType5/Coordinates"][1,:]
    r = radial_separation(pos, bh_pos_1)
    mask = np.logical_and(r < r_boundary[1], r > r_boundary[0])
    print(f"Inner {parttype} particles: {r[mask].shape[0]}")
    print(f"Outer {parttype} particles: {r[~mask].shape[0]}")
    selected_part_idx = rng.choice(np.arange(pos.shape[0])[mask])
    print(f"Selected {parttype} ID: {f[f'/{particles[parttype]}/ParticleIDs'][selected_part_idx]}")
    print("Distance to SMBHs: ")
    print(f"  SMBH 1: {r[selected_part_idx]}")
    r2 = radial_separation(pos[selected_part_idx, :], bh_pos_2)
    print(f"  SMBH 2: {r2.flatten()}")
    if r2 < r_boundary[0]:
        print(f"Particle to be perturbed is within {r_boundary[1]} kpc of SMBH 2! Cancelling...")
        quit()
    print("Before perturbing: ")
    print(f[f"/{particles[parttype]}/Coordinates"][selected_part_idx,:])
    f[f"/{particles[parttype]}/Coordinates"][selected_part_idx,0] += perturbation

with h5py.File(snapfile, "r") as f:
    print("After perturbing: ")
    print(f[f"/{particles[parttype]}/Coordinates"][selected_part_idx,:])