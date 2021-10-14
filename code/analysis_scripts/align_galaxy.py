import argparse
import numpy as np
import os
import pygad

parser = argparse.ArgumentParser(description="Align a snapshot with either the semiminor axis of the reduced inertia tensor or angular momentum of the stars, and save the snapshot as a new file.", allow_abbrev=False)
parser.add_argument(type=str, help="path to snapshot", dest="snap")
parser.add_argument("-o", "--orientate", type=str, help="how to orientate snapshot", dest="orientate", choices=["ri", "L"], default="ri")
parser.add_argument("-n", "--normal", type=str, help="vector normal to merger plane", dest="normal", choices=["x", "y", "z"], default="y")
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

#orientate the snapshot
#note this will orientate the RIT or L with the z axis!!
if args.orientate == "ri":
    mode = "red I"
else:
    mode = "L"
pygad.analysis.orientate_at(snap.stars, mode=mode, total=True)

#rotate the snapshot
rotation = pygad.transformation.rot_to_z(normal_vec)
rotation.apply(snap)
new_file = os.path.splitext(args.snap)[0]+"_aligned.hdf5"
snap.write(new_file)