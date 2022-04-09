import os
import multiprocessing as mp
import numpy as np
import cm_functions as cmf
import pygad

def run(i):
    if True:
        cdc = cmf.analysis.ChildSimData.load_from_file(cubefiles[i])
        snaplist = cmf.utils.get_snapshots_in_dir(perturbdirs[i])
        ballmask = pygad.BallMask(pygad.UnitScalar(30, "kpc"))
        num_vescs = np.full_like(snaplist, np.nan, dtype=float)
        jj = 0
        prev_hypers = []
        for j, snapfile in enumerate(snaplist):
            try:
                snap = pygad.Snapshot(snapfile, physical=True)
            except KeyError:
                continue
            snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
            snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")
            num_vescs[jj], prev_hypers = cmf.analysis.count_new_hypervelocity_particles(snap[ballmask], prev=prev_hypers)
            # save memory
            snap.delete_blocks()
            jj += 1
        # add to file
        cdc.update_hdf5_field("num_escaping_stars", num_vescs, "/galaxy_properties")
        print(f"Thread {i} updated file {cubefiles[i]} successfully!")


if __name__ == "__main__":
    maindir = "/scratch/pjohanss/arawling/collisionless_merger/mergers"
    root, cubes,_ = next(os.walk(os.path.join(maindir, "cubes")))
    cubedirs = [os.path.join(root, c) for c in cubes]
    cubedirs.sort()
    root, data, _ = next(os.walk(maindir))
    data.remove("cubes")
    data.remove("cubes_old")
    datadirs = [os.path.join(root, d, "perturbations") for d in data]
    datadirs.sort()
    pids = np.arange(10)
    for k, (cube, datadir) in enumerate(zip(cubedirs, datadirs)):
        root,_,_cubefiles = next(os.walk(cube))
        cubefiles = [os.path.join(root, c) for c in _cubefiles]
        cubefiles.sort()
        root, _pdirs,_ = next(os.walk(datadir))
        perturbdirs = [os.path.join(root, p, "output") for p in _pdirs]
        perturbdirs.sort()
        with mp.Pool(processes=len(pids)) as pool:
            pool.map(run, pids)
    print("Done")