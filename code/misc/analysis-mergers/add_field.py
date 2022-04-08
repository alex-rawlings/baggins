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
        for j, snapfile in enumerate(snaplist):
            snap = pygad.Snapshot(snapfile, physical=True)
            snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
            snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")
            vesc_func = cmf.analysis.escape_velocity(snap[ballmask])
            vmag = pygad.utils.geo.dist(snap.stars[ballmask])
            num_vescs[j] = np.sum(vmag>vesc_func(snap.stars[ballmask]["r"]))
            # save memory
            snap.delete_blocks()
        # add to file
        cdc.add_hdf5_field("num_escaping_stars", num_vescs, "/galaxy_properties")
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
        with mp.Pool(processes=10) as pool:
            pool.map(run, pids)
    print("Done")