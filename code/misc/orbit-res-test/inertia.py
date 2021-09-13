import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf

#load the snap
#snapfile = '/Volumes/Rawlings_Storage/KETJU/data/merger/P03_P03/data/P03_P03_o95_000.hdf5'
snapfile = '/Volumes/Rawlings_Storage/KETJU/initialise/merger/NGCa0524b-NGCa3348-2.0-0.001.hdf5'
#snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/fiducial/0-001/NGCa0524b-NGCa3348-2.0-0.001.hdf5'
#snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/reduced-x-10/0-001/output/0001_031.hdf5'
#snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/reduced-x-05/NGCa0524b/NGCa0524b.hdf5'

savedir = '/users/arawling/figures/res-test'
family = 'dm'
mask_type = 'shell'

ratios = np.full((10,2), np.nan)
particle_counts = np.full(10, np.nan)
radii = np.linspace(10, 5000, 10)

if False:
    #test merger-ic-generator
    snapfile = '/Volumes/Rawlings_Storage/KETJU/initialise/galaxies/extended_halo/NGCa3348/NGCa3348_009.hdf5'
    if False:
        #test an isolated galaxy
        import merger_ic_generator as mg
        isogal1 = mg.SnapshotSystem(snapfile)
        test_file_name = '/Volumes/Rawlings_Storage/KETJU/initialise/test/test.hdf5'
        mg.write_hdf5_ic_file(test_file_name, isogal1)
        snap = pygad.Snapshot(test_file_name)
        snap.to_physical_units()
        xcom = cmf.analysis.get_com_of_each_galaxy(snap)
        vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom)

        for i, (ri, ro) in enumerate(zip(radii[:-1], radii[1:])):
            print(xcom)
            for i2, d in enumerate(xcom.keys()):
                dm_radial_mask = cmf.analysis.get_radial_mask(snap, radius=[ri, ro], centre=xcom[d], family='dm')
                this_snap = snap.dm[dm_radial_mask]
                ratios[i,:] =  cmf.analysis.get_galaxy_axis_ratios(this_snap, xcom[d], vcom[d], family=family)
                print(ratios[i,:])
    else:
        #test a merger
        import merger_ic_generator as mg
        isogal1 = mg.SnapshotSystem(snapfile)
        snapfile2 = '/Volumes/Rawlings_Storage/KETJU/initialise/galaxies/extended_halo/NGCa0524b/NGCa0524b_009.hdf5'
        isogal2 = mg.SnapshotSystem(snapfile2)
        merger = mg.Merger(isogal1, isogal2, 5000, 1)
        test_file_name = '/Volumes/Rawlings_Storage/KETJU/initialise/test/test.hdf5'
        mg.write_hdf5_ic_file(test_file_name, merger)
        snap = pygad.Snapshot(test_file_name)
        snap.to_physical_units()

        fig, ax = plt.subplots(1,2, figsize=(7,4))
        pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax[0], extent=75000)
        ax[0].scatter(snap.bh['pos'][:,0], snap.bh['pos'][:,2], c='black', zorder=20)

        star_id_masks = cmf.analysis.get_all_id_masks(snap)
        dm_id_masks = cmf.analysis.get_all_id_masks(snap, family='dm')
        print('ID Lengths')
        for i1, idx in enumerate(dm_id_masks.keys()):
            print("{}: {}".format(idx, len(snap.dm[dm_id_masks[idx]])))
            print("BH Mass: {}".format(snap.bh[snap.bh['ID']==idx]['mass']))
        xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks)
        vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=star_id_masks)
        for idx in xcom.keys():
            ax[0].scatter(xcom[idx][0], xcom[idx][2], c='tab:red', zorder=20)

        print(xcom)
        for i, (ri, ro) in enumerate(zip(radii[:-1], radii[1:])):
            for i2, d in enumerate(xcom.keys()):
                dm_radial_masks = cmf.analysis.get_all_radial_masks(snap, radius=[ri, ro], family='dm', id_masks=dm_id_masks)
                if i2==0: continue
                print("xcom: {}".format(xcom[d]))
                print("BH pos: {}".format(snap.bh[snap.bh['ID']==d]['pos']))
                ratios[i,:] =  cmf.analysis.get_galaxy_axis_ratios(snap, xcom[d], family=family, radial_mask=dm_radial_masks[d])
                print(snap.dm['pos'][0,:])
                print(ratios[i,:])
    labvals = ['b/a', 'c/a']
    for j in range(2):
        ax[1].plot(radii, ratios[:,j], '-o', label=labvals[j])
    ax[1].set_xlabel('Radius/kpc')
    ax[1].set_ylabel('Axis ratio')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    quit()

snap = pygad.Snapshot(snapfile)
snap.to_physical_units()

star_id_masks = cmf.analysis.get_all_id_masks(snap)
dm_id_masks = cmf.analysis.get_all_id_masks(snap, family='dm')
print('determining coms')
xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks)
vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=star_id_masks)

if mask_type == 'ball':
    for i, r in enumerate(radii):
        #create the ID mask
        print('creating masks')
        if family == 'stars':
            star_radial_mask = cmf.analysis.get_all_radial_masks(snap, radius=r, centre=xcom, id_masks=star_id_masks)
        elif family == 'dm':
            dm_radial_mask = cmf.analysis.get_all_radial_masks(snap, radius=r, centre=xcom, id_masks=dm_id_masks, family='dm')
        for ind, d in enumerate(xcom.keys()):
            if ind > 0 : continue
            print('BH: {}'.format(d))
            if family == 'stars':
                this_snap = snap.stars[star_radial_mask[d]]
            elif family == 'dm':
                this_snap = snap.dm[dm_radial_mask[d]]
            else:
                raise ValueError('must be stars or dm')
            particle_counts[i] = len(this_snap)
            ratios[i,:] = cmf.analysis.get_galaxy_axis_ratios(this_snap, xcom[d], vcom[d], family=family)
            print(ratios[i,:])
elif mask_type == 'shell':
    for i, (ri, ro) in enumerate(zip(radii[:-1], radii[1:])):
        star_radial_mask = cmf.analysis.get_all_radial_masks(snap, radius=[ri, ro], centre=xcom, id_masks=star_id_masks)
        dm_radial_mask = cmf.analysis.get_all_radial_masks(snap, family='dm', radius=[ri, ro], centre=xcom, id_masks=dm_id_masks)
        for ind, d in enumerate(xcom.keys()):
            if ind > 0 : continue
            print('BH: {}'.format(d))
            if family == 'stars':
                this_snap = snap.stars[star_radial_mask[d]]
            elif family == 'dm':
                this_snap = snap.dm[dm_radial_mask[d]]
            else:
                raise ValueError('must be stars or dm')
            particle_counts[i] = len(this_snap)
            ratios[i,:] = cmf.analysis.get_galaxy_axis_ratios(this_snap, xcom[d], vcom[d], family=family)
            print(ratios[i,:])
labvals = ['b/a', 'c/a']
for i in range(2):
    plt.plot(radii, ratios[:,i], '-o', label=labvals[i])
#plt.ylim(0,1)
plt.legend()
plt.xlabel('Radii [kpc]')
plt.ylabel('Axis ratio from inertia tensor of particles within radius r')
plt.show()
#plt.savefig('{}/{}_axis_ratio_{}'.format(savedir, mask_type, family), dpi=300)
#plt.close()

plt.plot(radii, particle_counts, '-o')
plt.xlabel('Radii [kpc]')
plt.ylabel('{} Counts'.format(family))
plt.tight_layout()
plt.show()
#plt.savefig('{}/{}_counts_{}.png'.format(savedir, mask_type, family), dpi=300)
#plt.close()

if family == 'stars':
    fig, (ax1, ax2) = plt.subplots(1,2, sharex='all', sharey='all')
    _,ax1,*_ = pygad.plotting.image(snap.stars, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1, extent=12000)
    for i, idx in enumerate(xcom.keys()):
        try:
            _,ax2,*_ = pygad.plotting.image(snap.stars[star_radial_mask[idx]], qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2, extent=12000)
        except IndexError:
            print('Index error whilst plotting masks...')
            continue
elif family == 'dm':
    fig, (ax1, ax2) = plt.subplots(1,2, sharex='all', sharey='all')
    _,ax1,*_ = pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1, extent=12000)
    for k in dm_radial_mask.keys():
        print('Plotting for key {}'.format(k))
        try:
            pygad.plotting.image(snap.dm[dm_radial_mask[k]], qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2, extent=12000)
        except IndexError:
            print('Index error whilst plotting masks...')
            continue
plt.tight_layout()
plt.show()
#plt.savefig('{}/{}_masks_{}.png'.format(savedir, mask_type, family), dpi=300)
#plt.close()