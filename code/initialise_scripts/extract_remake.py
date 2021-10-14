import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pygad
import ketjugw
import merger_ic_generator as mg
import cm_functions as cmf


#get the command line options
parser = cmf.utils.argparse_for_initialise(description="Extract and regenerate a system.")
parser.add_argument('-p', '--plot', dest='plot', help='plot the merger setup', action='store_true')
args = parser.parse_args()

#get the parameter file values
pfv = cmf.utils.read_parameters(args.paramFile, verbose=args.verbose)

assert pfv.regeneration
#extract the galaxy at first pericentre
#first get a list of all snapshots
snap_path = os.path.join(pfv.saveLocation, "output/")
snap_files = cmf.utils.get_snapshots_in_dir(snap_path)
# TODO need to rename bh file to prevent overwriting
bh_file = os.path.join(snap_path, "ketju_bhs.hdf5")
bh1, bh2 = ketjugw.data_input.load_hdf5(bh_file).values()
#determine which snap to extract
pericentre_times, peri_idx = cmf.analysis.find_pericentre_time(bh1, bh2)
snap_to_extract = cmf.analysis.snap_num_for_time(snap_files, pericentre_times[0])
if args.verbose:
    print("Extracting from snap: {}".format(snap_files[snap_to_extract]))
snap = pygad.Snapshot(snap_files[snap_to_extract])
snap.to_physical_units()
#get the ID mask
star_id_masks = cmf.analysis.get_all_id_masks(snap)
#get the CoM positions and velocities
xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks, verbose=args.verbose)
vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=star_id_masks, verbose=args.verbose)
if args.verbose:
    print("CoM positions: \n  {}".format(xcom.values()))
    print("CoM velocities: \n  {}".format(vcom.values()))
snap.delete_blocks()
ordered_bh_key = list(xcom.keys())
ordered_bh_key.sort()
system1 = mg.SnapshotSystem(pfv.fileHigh1)
system2 = mg.SnapshotSystem(pfv.fileHigh2)
highres_system = mg.TwoBodySystem(system1, system2, xcom[ordered_bh_key[0]].view(np.ndarray), xcom[ordered_bh_key[1]].view(np.ndarray), vcom[ordered_bh_key[0]].view(np.ndarray), vcom[ordered_bh_key[1]].view(np.ndarray))
highres_name = os.path.join(pfv.saveLocation, "{}{}H.hdf5".format(pfv.galaxyName1, pfv.galaxyName2))
if args.verbose:
    print("Writing new file: {}".format(highres_name))
mg.write_hdf5_ic_file(highres_name, highres_system)

if args.plot:
    #plot the setup
    if args.verbose:
        print('Plotting...')
    snap = pygad.Snapshot(highres_name)
    snap.to_physical_units()
    cmf.plotting.plot_galaxies_with_pygad(snap, extent={"stars":400, "dm":5000})
    #plt.savefig('{}.png'.format(save_file_as.split('.')[0]), dpi=300)
    plt.show()