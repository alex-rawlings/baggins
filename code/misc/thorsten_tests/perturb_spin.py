import os
import numpy as np
import scipy.stats
import h5py
import cm_functions as cmf


datadir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/remnant_perturbations/009"

rng = np.random.default_rng(42)
CONST_G = 43007.1
CONST_c = 2.99792458e5

SL = cmf.CustomLogger("script", console_level="INFO")

subdirs = [f.path for f in os.scandir(datadir) if f.is_dir()]
subdirs.sort()

for j, s in enumerate(subdirs):
    if j<1: continue
    ic_file = cmf.utils.get_snapshots_in_dir(s)
    try:
        assert len(ic_file) < 2
    except AssertionError:
        SL.logger.exception("More than one snapshot returned! Unable to determine which is the IC file.")
        raise
    # determine new spins of BHs
    spin_params = cmf.literature.zlochower_dry_spins
    spin_mag = scipy.stats.beta.rvs(*spin_params.values(), size=2, random_state=rng)
    theta, phi = cmf.mathematics.uniform_sample_sphere(2, rng=rng)
    # assign new spins
    with h5py.File(ic_file[0], "r+") as f:
        spins = f["/PartType5/Spins"][:]
        masses = f["/PartType5/Masses"][:]
        SL.logger.info(f"Original spins:\n{spins}")
        for i, (m,s,t,p) in enumerate(zip(masses,spin_mag,theta,phi)):
            spins[i,:] = s * np.array([
                                        np.sin(t) * np.cos(p),
                                        np.sin(t) * np.sin(t),
                                        np.cos(t)
                                    ]) * CONST_G/CONST_c * m**2
            try:
                spin_norm = np.linalg.norm(spins[i,:])
                assert spin_norm < CONST_G/CONST_c * m**2
            except AssertionError:
                SL.logger.exception(f"BH {i} has spin norm of {spin_norm}!")
                raise
        # uncomment below line to update hdf5 fields
        f["/PartType5/Spins"][:] = spins
        SL.logger.info(f"New spins:\n{f['/PartType5/Spins'][:]}")