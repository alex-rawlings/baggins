import argparse
from cm_functions.utils.data_handling import save_data
import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


#set up the command line options
parser = argparse.ArgumentParser(description='Determine radially-dependent axis ratios for a range of snapshots', allow_abbrev=False)
parser.add_argument(type=str, help='path to snapshots or data', dest='path')
parser.add_argument('-n', '--new', dest='new', action='store_true', help='analyse a new dataset')
parser.add_argument('-m', '--method', type=str, help='method of determining inertia tensor', dest='method', choices=['shell', 'ball'], default='shell')
parser.add_argument('-f', '--family', type=str, help='particle family', dest='family', choices=['dm', 'stars'], default='dm')
parser.add_argument('-S', '--statistic', type=str, help='statistic', dest='stat', choices=['median', 'mean', 'last'], default='median')
parser.add_argument('-r', '--radii', type=cmf.utils.cl_str_2_space, help='radii to calculate inertia tensor at', dest='radii', default=np.linspace(0.1, 6, 20))
parser.add_argument('-s', '--savedir', type=str, help='save directory', dest='savedir', default='/users/arawling/figures/res-test/inertia')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose printing in script')
args = parser.parse_args()

if args.new:
    #analyse a new dataset
    #get the full pathname of all the snapshots
    snap_files = cmf.utils.get_snapshots_in_dir(args.path)

    #instantiate arrays
    time_of_snap = np.full_like(snap_files, np.nan, dtype=float)
    ratios = dict(
        galA = np.full((len(snap_files), 2), np.nan, dtype=float),
        galB = np.full((len(snap_files), 2), np.nan, dtype=float)
    )
    ratio_errors = dict(
        galA = np.full((len(snap_files), 4), np.nan, dtype=float),
        galB = np.full((len(snap_files), 4), np.nan, dtype=float)
    )
    particle_counts = dict(
        galA = np.full_like(snap_files, np.nan, dtype=float),
        galB = np.full_like(snap_files, np.nan, dtype=float)
    )

    for ind, this_file in enumerate(snap_files):
        print('Reading: {}'.format(this_file))

        #load the snapshot
        snap = pygad.Snapshot(this_file)
        snap.to_physical_units()
        time_of_snap[ind] = cmf.general.convert_gadget_time(snap)

        #mask the galaxies by ID and determine centre of mass positions
        #this only needs to be done on the first iteration
        if ind == 0:
            if args.verbose:
                print('Creating ID masks...')
            id_masks = dict(
                stars = cmf.analysis.get_all_id_masks(snap),
                dm = cmf.analysis.get_all_id_masks(snap, family='dm')
            )
            xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=id_masks['stars'], verbose=args.verbose)
            #determine the larger virial radius
            virial_radius, virial_mass = cmf.analysis.get_virial_info_of_each_galaxy(snap, xcom=xcom, masks=[id_masks['stars'], id_masks['dm']])
            virial_keys = list(virial_radius.keys())
            if virial_radius[virial_keys[0]] > virial_radius[virial_keys[1]]:
                args.radii *= virial_radius[virial_keys[0]]
            else:
                args.radii *= virial_radius[virial_keys[1]]
            #determine how to iterate over ball or shell
            if args.method == 'ball':
                radii_to_mask = args.radii
            else:
                radii_to_mask = list(zip(args.radii[:-1], args.radii[1:]))
        else:
            xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=id_masks['stars'], verbose=args.verbose)

        #set up temporary holding dicts
        temp_ratio = dict(
            galA = np.full((len(radii_to_mask), 2), np.nan, dtype=float),
            galB = np.full((len(radii_to_mask), 2), np.nan, dtype=float)
        )
        temp_partcount = dict(
            galA = np.full_like(radii_to_mask, np.nan, dtype=float),
            galB = np.full_like(radii_to_mask, np.nan, dtype=float)
        )
        #iterate over the radii to investigate
        for ind2, r in enumerate(radii_to_mask):
            if args.verbose:
                print('Creating radial masks...')
            radial_masks = cmf.analysis.get_all_radial_masks(snap, radius=r, 
                centre=xcom, id_masks=id_masks[args.family])
            #iterate over each progenitor
            for ind3, (key_x, key_r) in enumerate(zip(xcom.keys(), ratios.keys())):
                if len(snap[radial_masks[key_x]]) < 5000:
                    #filter out low bin counts
                    continue
                temp_ratio[key_r][ind2, :] = cmf.analysis.get_galaxy_axis_ratios(
                    snap, xcom[key_x], family=args.family, radial_mask=radial_masks[key_x]
                    )
                temp_partcount[key_r][ind2] = len(snap[radial_masks[key_x]])
        #we have now iterated over all radii
        #choose statistic we want use
        for key_r in ratios.keys():
            if args.stat == 'median':
                ratios[key_r][ind, :] = np.nanmedian(temp_ratio[key_r], axis=0)
                ratio_errors[key_r][ind, [0,2]] = ratios[key_r][ind, :] - np.nanquantile(temp_ratio[key_r], 0.25, axis=0)
                ratio_errors[key_r][ind, [1,3]] = np.nanquantile(temp_ratio[key_r], 0.75, axis=0) - ratios[key_r][ind, :]
                particle_counts[key_r][ind] = np.nanmedian(temp_partcount[key_r])
            elif args.stat == 'mean':
                ratios[key_r][ind, :] = np.nanmean(temp_ratio[key_r], axis=0)
                st_devs = np.nanstd(temp_ratio[key_r], axis=0)
                ratio_errors[key_r][ind, [0,2]] = st_devs
                ratio_errors[key_r][ind, [1,3]] = -st_devs
                particle_counts[key_r][ind] = np.nanmean(temp_partcount[key_r])
            else:
                ratios[key_r][ind, :] = temp_ratio[key_r][:,-1]
                particle_counts[key_r][ind] = temp_partcount[key_r][-1]
        snap.delete_blocks()
        #delete the snapshot to preserve memory
        del snap
        print('\n')
    save_dict = dict(
        time_of_snap = time_of_snap,
        ratios = ratios,
        ratio_errors = ratio_errors,
        xcom = xcom,
        particle_counts = particle_counts
    )
    
    input_name_split = args.path.strip('/').split('/')
    savefile_name = "inertia-{}-{}".format(input_name_split[-3], input_name_split[-2])
    cmf.utils.save_data(save_dict, "{}.pickle".format(savefile_name))
else:
    #read in a new dataset
    if args.verbose:
        print('Loading saved dataset...')
    savefile_name = args.path.lstrip('./').split('.')[-2]
    save_dict = cmf.utils.load_data(args.path)
    time_of_snap = save_dict['time_of_snap']
    ratios = save_dict['ratios']
    ratio_errors = save_dict['ratio_errors']
    xcom = save_dict['xcom']
    particle_counts = save_dict['particle_counts']


#plot
fig, ax = plt.subplots(3, 1, figsize=(6,6), sharex='all')
labvals = ['b/a', 'c/a']
for (key_r, key_x) in zip(ratios.keys(), xcom.keys()):
    #need to transpose ratio errors for pyplot compatability
    this_ratio_errors = ratio_errors[key_r].T
    ax[0].errorbar(time_of_snap, ratios[key_r][:,0], yerr=this_ratio_errors[:2, :], label=key_x)
    ax[1].errorbar(time_of_snap, ratios[key_r][:,1], yerr=this_ratio_errors[2:, :])
    ax[2].plot(time_of_snap, particle_counts[key_r], '-o')
for axi in (ax[0], ax[1]):
    axi.axhline(0.9, c='k', alpha=0.4)
ax[2].set_xlabel('Time [Gyr]')
ax[0].legend()
ax[0].set_ylabel('b/a')
ax[1].set_ylabel('c/a')
ax[2].set_ylabel('Particle Count')
plt.tight_layout()
#plt.savefig('{}/{}.png'.format(args.savedir, savefile_name), dpi=300)
plt.show()
