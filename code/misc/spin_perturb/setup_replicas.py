import os
import shutil
import numpy as np
import ketjugw
import h5py
import merger_ic_generator as mig
from baggins.mathematics import uniform_sample_sphere, convert_spherical_to_cartesian

# set input
a0 = 0.5
ecc = 0.9
mbh = 1e9
dir_exist_ok = False
random_dir = False


output_path = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/spin_test/shmem_preserve/compare_spins_fixed_dir/{a0:.1f}pc"
gadget = 1e10
kpc = 1e3 * ketjugw.units.pc


# ok, let's make the Keplerian binary that we want: then we can give the pos 
# and vel to the Snapshot
_bh1, _bh2 = ketjugw.keplerian_orbit(mbh, mbh, a0*ketjugw.units.pc, ecc, l=0, ts=0)
r0 = float(np.linalg.norm(_bh1.x-_bh2.x) / kpc)
rperi = float(a0*(1-ecc) / 1e3)
print("Target coordinates:")
for _bh in (_bh1, _bh2):
    print(f"  pos: {_bh.x/ketjugw.units.pc}")
    print(f"  vel: {_bh.v/ketjugw.units.km_per_s}")

# set up the system
# set up the stars
stars = mig.DehnenSphere(1e11/gadget, 3, 1, 2e5/gadget, mig.ParticleType.STARS)

# set up the BH
bh = mig.CentralPointMass(mbh/gadget, mig.ParticleType.BH)
galaxy = mig.ErgodicSphericalSystem(stars, bh)
galaxy = mig.TransformedSystem(galaxy, mig.FilterParticlesBoundToCentralMass(central_object_mass=mbh, minimum_semi_major_axis=1e-3))

# merger between a standard galaxy and a single BH
merger = mig.Merger(galaxy, mig.ErgodicSphericalSystem(bh), r0=r0, rperi=rperi, e=ecc, mass_radius_fac=1e-50)
os.makedirs(output_path, exist_ok=dir_exist_ok)
mig.write_hdf5_ic_file(os.path.join(output_path, "system.hdf5"), merger)

# read in file for editing
spin_dict = {"A":0, "B":0.3, "C":0.5, "D":0.9}
for l1, chi1 in spin_dict.items():
    for l2, chi2 in spin_dict.items():
        # copy file
        os.makedirs(os.path.join(output_path, f"{l1}{l2}"), exist_ok=dir_exist_ok)
        newfile = shutil.copy(os.path.join(output_path, "system.hdf5"), os.path.join(output_path, f"{l1}{l2}/{l1}{l2}.hdf5"))
        print(f"Combination to make: {l1}{l2}")
        with h5py.File(newfile, "r+") as f:
            for i, (_bh, chi) in enumerate(zip((_bh1, _bh2), (chi1, chi2))):
                # update positions and velocities
                # let's reduce the velocities so the BHs don't fly off
                f["/PartType5/Coordinates"][i,:] = _bh.x[0]/ketjugw.units.pc
                f["/PartType5/Velocities"][i,:] = _bh.v[0]/ketjugw.units.km_per_s/1e3
                # update spins
                if random_dir:
                    # orientate randomly
                    t, p = uniform_sample_sphere(1)
                    spin = convert_spherical_to_cartesian([chi, t[0], p[0]])
                else:
                    spin = np.array([0, chi*(-1)**i, 0])
                f["/PartType5/Spins"][i,:] = spin.flatten() * 43007.1 / 2.99792458e5 * f["/PartType5/Masses"][i]**2

        # check update was successful
        if False:
            with h5py.File(newfile, "r") as f:
                for i, _bh in enumerate((_bh1, _bh2)):
                    assert np.allclose(f["/PartType5/Coordinates"][i,:], _bh.x[0]/ketjugw.units.pc)
                    assert np.allclose(f["/PartType5/Velocities"][i,:], _bh.v[0]/ketjugw.units.km_per_s)