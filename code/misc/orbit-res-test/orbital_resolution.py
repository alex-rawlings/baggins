import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os
import pygad
import ketjugw
import baggins as bgs


parser = argparse.ArgumentParser(description='Visualise the error in orbital trajectory arising from differeing mass resolutions.', allow_abbrev=False)
parser.add_argument('-d', '--data', type=str, help='path to simulation data', dest='path', default=None)
parser.add_argument('-l', '--load', type=str, help='file to load', dest='load', default=None)
parser.add_argument('-p', '--projection', type=int, default=1, choices=[0,1,2], help='Viewing projection', dest='proj')
parser.add_argument('-o', '--orbit', type=str, help='Orbital approach type', dest='orbit', default=None)
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose printing', dest='verbose')
args = parser.parse_args()


if args.path is not None:
    #we want to read in a new dataset
    need_bh_masses = True
    for ind, (root, directories, files) in enumerate(os.walk(args.path)):
        if ind == 0:
            #we are in the top directory
            data_dict = dict()
            data_dict['bh_masses'] = np.full(2, np.nan)
            initial_array = np.empty((0,3), float)
            for d in directories:
                data_dict[d] = dict(
                time = np.empty(0, float),
                xcom = {'0':initial_array, '1':initial_array},
                vcom = {'0':initial_array, '1':initial_array},
                bhx = {'0':initial_array, '1':initial_array},
                )
        else:
            if args.orbit is not None and args.orbit not in root:
                #this isn't the orbit type we want
                continue
            #we need to know wich resolution we are dealing with
            resolution = [r for r in list(data_dict.keys()) if r in root][0]

            for ind2, file_ in enumerate(files):
                if file_.endswith('.hdf5') and 'ketju_bhs' not in file_:
                    full_path_to_file = os.path.join(root, file_)
                    print('Reading: {}'.format(full_path_to_file))
                    snap = pygad.Snapshot(full_path_to_file)
                    snap.to_physical_units()
                    #convention! com1 is the smaller ID number
                    #an com2 is the larger ID number
                    bh_id_order = np.argsort(snap.bh['ID'])
                    if need_bh_masses:
                        #get the BH masses
                        data_dict['bh_masses'] = snap.bh['mass'][bh_id_order]
                        need_bh_masses = False
                    if args.orbit is not None and root.split('/')[-1] == args.orbit:
                        #this is the ic file
                        id_masks_for = resolution
                        id_masks = bgs.analysis.get_all_id_masks(snap)
                        data_dict[resolution]['time'] = np.hstack([data_dict[resolution]['time'], [0]])
                    else:
                        #this is a regular file
                        data_dict[resolution]['time'] = np.hstack([data_dict[resolution]['time'],
                                bgs.general.convert_gadget_time(snap)
                                ])
                    #sanity check
                    if id_masks_for != resolution:
                        raise RuntimeError('Need the correct ID mask!')
                    #get the data
                    xcoms = bgs.analysis.get_coms_of_each_galaxy(snap, masks=id_masks, verbose=args.verbose)
                    vcoms = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcoms, masks=id_masks, verbose=args.verbose)
                    #and assign to the dictionary
                    bh_id_key = snap.bh['ID'][bh_id_order]
                    for subkey in range(2):
                        data_dict[resolution]['xcom'][str(subkey)] = np.vstack([
                                data_dict[resolution]['xcom'][str(subkey)],
                                xcoms[bh_id_key[subkey]]
                        ])
                        data_dict[resolution]['vcom'][str(subkey)] = np.vstack([
                                data_dict[resolution]['vcom'][str(subkey)],
                                vcoms[bh_id_key[subkey]]
                        ])
                        data_dict[resolution]['bhx'][str(subkey)] = np.vstack([
                                data_dict[resolution]['bhx'][str(subkey)],
                                snap.bh['pos'][bh_id_order[subkey],:]
                        ])
                    snap.delete_blocks()
    #sort the data based on time
    if args.verbose:
        print('Sorting...')
    for d in list(data_dict.keys()):
        if d == 'bh_masses':
            continue
        sorted_idx = np.argsort(data_dict[d]['time'])
        for sub_d in data_dict[d]:
            if sub_d == 'time':
                data_dict[d][sub_d] = data_dict[d][sub_d][sorted_idx]
            else:
                for i in range(2):
                    data_dict[d][sub_d][str(i)] = data_dict[d][sub_d][str(i)][sorted_idx, :]

    #save data
    if args.orbit is not None:
        bgs.utils.save_data(data_dict, '{}.pickle'.format(args.orbit))
    else:
        bgs.utils.save_data(data_dict, 'general-res-test.pickle')
else:
    #we want to load some previous data set
    if args.load is None:
        raise ValueError('A file must be specified for reading previous data!')
    data_dict = bgs.utils.load_data(args.load)
    print('Data {} loaded'.format(args.load))

    save_dir = os.path.join('/users/arawling/figures/res-test/', args.load.split('.')[0])
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass
    axes = [i for i in range(3) if i != args.proj]
    axes_labels = {'xcom':['x/kpc', 'y/kpc', 'z/kpc'],
                    'vcom':[r'v$_\mathrm{x}$/km/s', 'v$_\mathrm{y}$/km/s', 'v$_\mathrm{z}$/km/s']}
    kpc = 1e3 * ketjugw.units.pc
    max_time = 1.4

    cols = bgs.plotting.mplColours()
    linestyles = ['-', ':', '--', '-.'] # list(bgs.plotting.mplLines().items())
    # TODO: create a marker list like above
    markers = ['o', 's', '^', 'v', 'D']
    nbins=8
    splinekind = 'quadratic'

    #convert to CoM frame
    xcom = dict()
    vcom = dict()
    for d in list(data_dict.keys()):
        if d == 'bh_masses':
            continue
        xcom[d] = (data_dict[d]['xcom']['0']*data_dict['bh_masses'][0] + data_dict[d]['xcom']['1']*data_dict['bh_masses'][1]) / np.sum(data_dict['bh_masses'])
        vcom[d] = (data_dict[d]['vcom']['0']*data_dict['bh_masses'][0] + data_dict[d]['vcom']['1']*data_dict['bh_masses'][1]) / np.sum(data_dict['bh_masses'])
        for i in range(2):
            data_dict[d]['xcom'][str(i)] -= xcom[d]
            data_dict[d]['vcom'][str(i)] -= vcom[d]
    resolutions = [r for r in list(data_dict.keys()) if r != 'bh_masses']

    #set up the figure
    for fi, com in enumerate(('xcom', 'vcom')):
        print('Plotting: {}'.format(com))
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel(axes_labels[com][axes[0]])
        ax.set_ylabel(axes_labels[com][axes[1]])
        ax.set_title('CoM in Coordinate Space')
        time_seq = np.linspace(0, np.min([data_dict[r]['time'][-1] for r in resolutions]), 2000)
        for ind, d in enumerate(resolutions):
            time_mask = time_seq < max_time
            for i in range(2):
                interpx = scipy.interpolate.interp1d(
                        data_dict[d]['time'],
                        data_dict[d][com][str(i)][:, axes[0]],
                        kind = splinekind
                )
                interpz = scipy.interpolate.interp1d(
                        data_dict[d]['time'],
                        data_dict[d][com][str(i)][:, axes[1]],
                        kind = splinekind
                )
                ax.plot(interpx(time_seq[time_mask]), interpz(time_seq[time_mask]), c=cols[ind], marker=markers[i], label=(d if i < 1 else ''), markevery=100)
        ax.legend()
        plt.tight_layout()
        plt.savefig('{}/{}_coords.png'.format(save_dir, com), dpi=300)

        fig, ax = plt.subplots(3,1, figsize=(4,6), sharex='all')
        ax[2].set_xlabel('t/Gyr')
        ax[0].set_ylabel(r'||CoM$_{1}$ - CoM$_{2}$||')
        ax[1].set_ylabel(r'|CoM$_{1}$ - CoM$_{2}$|$_\mathrm{x}$')
        ax[2].set_ylabel(r'|CoM$_{1}$ - CoM$_{2}$|$_\mathrm{z}$')
        ax[0].set_title(r'CoM Differences')
        ax[1].set_yscale('log')
        ax[2].set_yscale('log')
        for ind, d in enumerate(resolutions):
            yinterp = scipy.interpolate.interp1d(data_dict[d]['time'],  np.sqrt(np.sum((data_dict[d][com]['0'] - data_dict[d][com]['1'])**2, axis=-1)), kind=splinekind)
            ax[0].plot(time_seq, yinterp(time_seq), label=d)
            for ind2 in range(2):
                yinterp2 = scipy.interpolate.interp1d(data_dict[d]['time'], np.abs(data_dict[d][com]['0'][:, axes[ind2]] - data_dict[d][com]['1'][:, axes[ind2]]), kind=splinekind)
                ax[ind2+1].plot(time_seq, yinterp2(time_seq), c=cols[ind], label=(d if ind2 < 1 else ''))
        ax[1].legend(fontsize='small')
        plt.tight_layout()
        plt.savefig('{}/{}_com_diff.png'.format(save_dir, com), dpi=300)

        fig, ax = plt.subplots(2,1, figsize=(7,5), sharex='all')
        ax[0].set_title('Difference to Fiducial')
        for axi in ax:
            axi.axhline(0, c='k', alpha=0.3)
        for ind, d in enumerate(resolutions):
            if d == 'fiducial':
                finterp = dict()
                for bh in ['0', '1']:
                    finterp[bh] = dict()
                    for coord in range(2):
                        finterp[bh][str(coord)] = scipy.interpolate.interp1d(data_dict[d]['time'],
                            data_dict[d][com][bh][:, axes[coord]], kind=splinekind)
            else:
                for bh in range(2):
                    for coord in range(2):
                        this_interp = scipy.interpolate.interp1d(data_dict[d]['time'],
                                data_dict[d][com][str(bh)][:, axes[coord]], kind=splinekind)
                        ax[coord].plot(time_seq, finterp[str(bh)][str(coord)](time_seq) - this_interp(time_seq), c=cols[ind-1], marker=markers[bh], label=(d if coord >0  and bh < 1 else ''), markevery=100)
        ax[1].set_xlabel('t/Gyr')
        ax[0].set_ylabel(axes_labels[com][axes[0]])
        ax[1].set_ylabel(axes_labels[com][axes[1]])
        ax[1].legend()
        fig.tight_layout()
        fig.savefig('{}/{}_fid_diff.png'.format(save_dir, com), dpi=300)
    #plt.show()
