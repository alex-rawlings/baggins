import warnings
import numpy as np
import scipy.stats
import os
import re
import shutil
import pygad
import cm_functions as cmf


#set up command line arguments
parser = cmf.utils.argparse_for_initialise(description="Generate a series of children runs which are perturbed versions of the parent.")
parser.add_argument("-e", "--exists", help="Allow for overwriting of existing directories", action="store_true", dest="exists")
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.paramFile)

#set up the RNG
rng = np.random.default_rng(pfv.seed)

#find the snapshot corresponding to the time we want
snaplist = cmf.utils.get_snapshots_in_dir(os.path.join(pfv.full_save_location, "output"))
pfv.perturb_snap_idx = cmf.analysis.snap_num_for_time(snaplist, pfv.perturbTime, units="Gyr", verbose=args.verbose)
snap = pygad.Snapshot(snaplist[pfv.perturb_snap_idx], physical=True)

#get com motions
star_id_masks = cmf.analysis.get_all_id_masks(snap)
xcoms = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks, verbose=args.verbose)
vcoms = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcoms, masks=star_id_masks, verbose=args.verbose)

#determine radial coordinate distribution in brownian motion
#and set up the perturbation values
perturb_dict = dict()
perturb_pos = dict()
perturb_vel = dict()
for (bhid, perturbfile) in zip(star_id_masks.keys(), (pfv.perturb1, pfv.perturb2)):
    perturb_dict[bhid] = cmf.utils.load_data(perturbfile)
    perturb_pos[bhid] = np.full((pfv.numberPerturbs, 3), np.nan, dtype=float)
    perturb_vel[bhid] = np.full((pfv.numberPerturbs, 3), np.nan, dtype=float)

for bhid in perturb_dict.keys():
    #get the magnitudes of the radial and velocity motions for KDE
    displacement = cmf.mathematics.radial_separation(perturb_dict[bhid]["diff_x"])
    vel_mag = cmf.mathematics.radial_separation(perturb_dict[bhid]["diff_v"])
    for (values, crd_dict, com_crd) in zip((displacement, vel_mag), (perturb_pos, perturb_vel), (xcoms, vcoms)):
        #perform the kernel density estimate, using a reflective boundary 
        #to prevent values < 0
        kernel = scipy.stats.gaussian_kde(values)
        samples = np.abs(kernel.resample(size=pfv.numberPerturbs, seed=rng))
        thetas, phis = cmf.mathematics.uniform_sample_sphere(pfv.numberPerturbs, rng)
        rtp = np.vstack((samples, thetas, phis)).T
        crd_dict[bhid] = cmf.mathematics.convert_spherical_to_cartesian(rtp) + com_crd[bhid]

#set up children directories and ICs
perturb_dir = os.path.join(pfv.full_save_location, "perturbations_eta_0005")
os.makedirs(perturb_dir, exist_ok=args.exists)
for i in range(pfv.numberPerturbs):
    print("Setting up child directories: {:.2f}%            ".format(i/(pfv.numberPerturbs-1)*100), end="\r")
    child_dir = os.path.join(perturb_dir, "{:03d}".format(i))
    os.makedirs(os.path.join(child_dir, "output"), exist_ok=args.exists)
    shutil.copyfile(os.path.join(pfv.full_save_location, "paramfile"), os.path.join(child_dir, "paramfile"))
    ic_file_name = "{}{}_perturb_{:03d}".format(pfv.galaxyName1, pfv.galaxyName2, i)
    shutil.copyfile(snaplist[pfv.perturb_snap_idx], os.path.join(child_dir, "{}.hdf5".format(ic_file_name)))
    #edit BH coordinates
    snap = pygad.Snapshot(os.path.join(child_dir, "{}.hdf5".format(ic_file_name)), physical=True)
    for bhid in snap.bh["ID"]:
        snap.bh["pos"][snap.bh["ID"] == bhid] = pygad.UnitArr(np.atleast_2d(perturb_pos[bhid][i,:]), units=snap["pos"].units)
        snap.bh["vel"][snap.bh["ID"] == bhid] = pygad.UnitArr(np.atleast_2d(perturb_vel[bhid][i,:]), units=snap["vel"].units)
    snap.write(os.path.join(child_dir, "{}.hdf5".format(ic_file_name)), overwrite=True, gformat=3)
    #edit paramfile
    with open(os.path.join(child_dir, "paramfile"), "r+") as f:
        contents = f.read()
        for param, val in zip(
            ("InitCondFile", "SnapshotFileBase", "SofteningStars", "ketju_disable_integration", "ErrTolIntAccuracy"), 
            (ic_file_name, ic_file_name, pfv.newStarSoftening, 0, 0.005)):
            line = re.search(r"^\b{}\b.*".format(param), contents, flags=re.MULTILINE)
            if line is None:
                warnings.warn("Parameter {} not in file! Skipping...".format(param))
                continue
            if "%" in line.group(0):
                comment = "  %" + "%".join(line.group(0).split("%")[1:])
            else:
                comment = ""
            contents, numsubs = re.subn(r"^\b{}\b.*".format(param), "{}  {}{}".format(param, val, comment), contents, flags=re.MULTILINE)
        f.seek(0)
        f.write(contents)
        f.truncate()
    
    
print("All child directories made.                                  ")
