import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pygad
import cm_functions as cmf


parser = argparse.ArgumentParser(description='Visualise the error in orbital trajectory arising from differeing mass resolutions.', allow_abbrev=False)
parser.add_argument(type=str, help='path to simulation', dest='path')
parser.add_argument('-p', '--projection', type=int, default=1, choices=[0,1,2], help='Viewing projection', dest='proj')
parser.add_argument('-o', '--orbit', type=str, help='Orbital approach type', dest='orbit')
args = parser.parse_args()

axes = [i for i in range(3) if i != args.proj]
#set up the figure
fig, ax = plt.subplots(1,1)
cols = cmf.plotting.mplColours()

snapcount = 0
for root, directories, files in os.walk(args.path):
    if args.orbit not in root:
        continue
    for ind, file_ in enumerate(files):
        if file_ == 'ketju_bhs.hdf5':
            full_path_to_file = os.path.abspath(os.path.join(root, file_))
            #copy file so we can open it if a simulation is still running
            full_path_to_new_file = full_path_to_file.split('.')[-2]+'_cp.hdf5'
            os.system('cp {} {}'.format(full_path_to_file, full_path_to_new_file))
            print('Reading from: {}'.format(full_path_to_new_file))
            bhs = ketjugw.data_input.load_hdf5(full_path_to_new_file)
            bh1, bh2 = bhs.values()
            #plot ketju bh output trajectories
            if bh1.m[-1] > bh2.m[-1]:
                ordered_bhs = (bh1, bh2)
            else:
                ordered_bhs = (bh2, bh1)
            for ind2, bh in enumerate(ordered_bhs):
                ax.plot(bh.x[:, axes[0]], bh.x[:, axes[1]], c=cols[ind2])
        elif file_.endswith('.hdf5'):
            #this is a snapshot
            full_path_to_file = os.path.abspath(os.path.join(root, file_))
            snap = pygad.Snapshot(full_path_to_file)
            snap.to_physical_units()
            if 'NGCa' in file_:
                #this is the IC file
                assert(snapcount == 0)
                print('Creating ID mask from initial condition file')
                id_masks = cmf.analysis.get_all_id_masks(snap)
            else:
                snapcount += 1
            coms = cmf.analysis.get_coms_of_each_galaxy(snap, masks=id_masks)
            for ind2 in range(2):
                ax.scatter(coms[ind2, axes[0]], coms[ind2, axes[1]], c=cols[ind2], zorder=10)
plt.show()
