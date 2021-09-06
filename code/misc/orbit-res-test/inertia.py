import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pygad
import cm_functions as cmf

#load the snap
snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/fiducial/0-001/NGCa0524b-NGCa3348-2.0-0.001.hdf5'
#snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/reduced-x-10/0-001/output/0001_031.hdf5'
#snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/reduced-x-05/NGCa0524b/NGCa0524b.hdf5'

savedir = '/users/arawling/figures/res-test'
family = 'dm'
mask_type = 'ball'

snap = pygad.Snapshot(snapfile)
snap.to_physical_units()

ratios = np.full((10,2), np.nan)
particle_counts = np.full(10, np.nan)
radii = np.linspace(10, 500, 10)

if False:
    snapfile = '/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa3607/output/NGCa3607_002.hdf5'
    snap = pygad.Snapshot(snapfile)
    snap.to_physical_units()
    xcom = cmf.analysis.get_com_of_each_galaxy(snap)
    print(xcom)
    vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom)
    for i, r in enumerate(radii):
        dm_radial_mask = cmf.analysis.get_radial_mask(snap, radius=r, centre=snap.bh['pos'], family='dm')
        #xcom[list(xcom.keys())[0]]
        this_snap = snap.dm[dm_radial_mask]
        for i2, d in enumerate(xcom):
            if i2>0: continue
            ratios[i,:] =  cmf.analysis.get_galaxy_axis_ratios(this_snap, xcom[d], vcom[d], family=family)
            print(ratios[i,:])
    quit()

star_id_masks = cmf.analysis.get_all_id_masks(snap)
dm_id_masks = cmf.analysis.get_all_id_masks(snap, family='dm')
print('determining coms')
xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks)
vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=star_id_masks)

"""
dm_radial_mask = cmf.analysis.get_all_radial_masks(snap, radius=10, centre='bh', id_masks=dm_id_masks, family='dm')
for k in dm_id_masks:
    plt.plot(snap.dm[dm_radial_mask[k]]['ID'], ls='', marker='.')
plt.show()
quit()
"""

if mask_type == 'ball':
    for i, r in enumerate(radii):
        #create the ID mask
        print('creating masks')
        star_radial_mask = cmf.analysis.get_all_radial_masks(snap, radius=r, centre=xcom, id_masks=star_id_masks)
        dm_radial_mask = cmf.analysis.get_all_radial_masks(snap, radius=r, centre=xcom, id_masks=dm_id_masks, family='dm')
        """fig, ax = plt.subplots(1,1)
        ax.set_aspect('equal')
        cols = ['tab:blue', 'tab:orange']
        for i2, idx in enumerate(dm_radial_mask.keys()):
            #if i2 > 0: continue
            plt.scatter(snap.dm[dm_radial_mask[idx]]['pos'][:,0], snap.dm[dm_radial_mask[idx]]['pos'][:,2], alpha=0.3, marker='.')
            bh = snap.bh[snap.bh['ID']==idx]['pos'][0]
            print(bh)
            plt.scatter(xcom[idx][0], xcom[idx][2], marker='s', c=cols[i2], zorder=10)
        plt.xlim(-2000, 2000)
        plt.ylim(-2000, 2000)"""
        for idx in xcom.keys():
            if family == 'stars':
                plt.plot(snap.stars[star_radial_mask[idx]]['ID'], snap.stars[star_radial_mask[idx]]['r'], ls='', marker='.')
            elif family == 'dm':
                plt.plot(snap.dm[dm_radial_mask[idx]]['ID'], snap.dm[dm_radial_mask[idx]]['r'], ls='', marker='.')
        plt.show()
        if i > 5:
            quit()
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
plt.savefig('{}/{}_axis_ratio_{}'.format(savedir, mask_type, family), dpi=300)
plt.close()

plt.plot(radii, particle_counts, '-o')
plt.xlabel('Radii [kpc]')
plt.ylabel('{} Counts'.format(family))
plt.tight_layout()
plt.savefig('{}/{}_counts_{}.png'.format(savedir, mask_type, family), dpi=300)
plt.close()

if family == 'stars':
    fig, (ax1, ax2) = plt.subplots(1,2, sharex='all', sharey='all')
    _,ax1,*_ = pygad.plotting.image(snap.stars, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1, extent=12000)
    for i, idx in enumerate(xcom.keys()):
        try:
            _,ax2,*_ = pygad.plotting.image(snap.stars[star_radial_mask[idx]], qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2, extent=12000)
        except IndexError:
            continue
elif family == 'dm':
    fig, (ax1, ax2) = plt.subplots(1,2, sharex='all', sharey='all')
    _,ax1,*_ = pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1, extent=12000)
    for i in range(2):
        try:
            _,ax2,*_ = pygad.plotting.image(snap.dm[dm_radial_mask[snap.bh['ID'][i]]], qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2, extent=12000)
        except IndexError:
            continue
plt.tight_layout()
plt.savefig('{}/{}_masks_{}.png'.format(savedir, mask_type, family), dpi=300)
plt.close()