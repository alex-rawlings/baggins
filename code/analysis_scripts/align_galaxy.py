import argparse
import numpy as np
import os
import warnings
import pygad
import cm_functions as cmf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Align a snapshot with either the semiminor axis of the reduced inertia tensor or angular momentum of the stars, and save the snapshot as a new file.", allow_abbrev=False)
parser.add_argument(type=str, help="path to snapshot", dest="snap")
parser.add_argument("-o", "--orientate", type=str, help="how to orientate snapshot", dest="orientate", choices=["ri", "L"], default="ri")
parser.add_argument("-n", "--normal", type=str, help="vector normal to merger plane", dest="normal", choices=["x", "y", "z"], default="y")
parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="plot the snap before and after rotation")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose printing in script")
args = parser.parse_args()


#set up the normal vectors
if args.normal == "x":
    normal_vec = np.array([1,0,0])
elif args.normal == "y":
    normal_vec = np.array([0,1,0])
else:
    normal_vec = np.array([0,0,1])

#load the snapshot
snap = pygad.Snapshot(args.snap)
snap.to_physical_units()
if args.plot:
    cmf.plotting.plot_galaxies_with_pygad(snap)
#orientate the snapshot
#note this will orientate the RIT or L with the z axis!!
if args.orientate == "ri":
    if args.verbose:
        print("Aligning snapshot with semiminor axis of the reduced inertia tensor...")
    mode = "red I"
else:
    if args.verbose:
        print("Aligning snapshot with the angular momentum vector...")
    mode = "L"
pygad.analysis.orientate_at(snap.stars, mode=mode, total=True)

#rotate the snapshot to the desired normal vector
rotation = pygad.transformation.rot_to_z(normal_vec)
rotation.apply(snap)

#recentre the snapshot
if args.verbose:
    print("Recentring galaxy wrt stellar CoM motion...")
xcom = cmf.analysis.get_com_of_each_galaxy(snap, verbose=args.verbose)
vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, verbose=args.verbose)
snap["pos"] -= list(xcom.values())[0]
snap["vel"] -= list(vcom.values())[0]

if args.plot:
    cmf.plotting.plot_galaxies_with_pygad(snap)
    plt.show()
new_file = os.path.splitext(args.snap)[0]+"_aligned.hdf5"
if os.path.exists(new_file):
    warnings.warn("File {} will be overwritten!".format(new_file))
#gformat=3 for hdf5 output
snap.write(new_file, overwrite=True, gformat=3)
if args.verbose:
    print("New file {} written".format(new_file))