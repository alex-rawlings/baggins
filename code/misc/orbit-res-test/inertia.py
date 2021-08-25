import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pygad
import cm_functions as cmf

#load the snap
snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/fiducial/0-001/NGCa0524b-NGCa3348-2.0-0.001.hdf5'
#snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/reduced-x-10/0-001/output/0001_031.hdf5'

savedir = '/users/arawling/figures/res-test'
family = 'stars'
mask_type = 'ball'

snap = pygad.Snapshot(snapfile)
snap.to_physical_units()

ratios = np.full((10,2), np.nan)
particle_counts = np.full(10, np.nan)
radii = np.linspace(10, 500, 10)

if mask_type == 'ball':
    for i, r in enumerate(radii):
        #create the ID mask
        print('creating masks')
        star_id_masks = cmf.analysis.get_all_id_masks(snap, radius=r)
        dm_id_masks = cmf.analysis.get_all_id_masks(snap, family='dm', radius=r)

        #get com coordinates
        print('determining coms')
        xcom = cmf.analysis.get_coms_of_each_galaxy(snap, masks=star_id_masks)
        vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=star_id_masks)

        for ind, d in enumerate(xcom):
            if ind > 0 : continue
            print('BH: {}'.format(d))
            if family == 'stars':
                com_snap = snap.stars[star_id_masks[d]]
            elif family == 'dm':
                com_snap = snap.dm[dm_id_masks[d]]
            else:
                raise ValueError('must be stars or dm')
            particle_counts[i] = len(com_snap)
            com_snap['pos'] = com_snap['pos'] - xcom[d]
            com_snap['vel'] = com_snap['vel'] - vcom[d]
            #create the reduced inertia tensor
            print('determing I')
            inertia_tensor = pygad.analysis.reduced_inertia_tensor(com_snap)
            print('getting eigenvalues')
            eigen_vals, eigen_vecs = scipy.linalg.eig(inertia_tensor)
            idx = np.argsort(eigen_vals)[::-1]
            eigen_vals = np.real(eigen_vals[idx])
            eigen_vecs = eigen_vecs[:, idx]
            abc = np.sqrt(eigen_vals)
            print('axes')
            print(abc)
            axis_ratios = abc[1:] / abc[0]
            print(axis_ratios)
            ratios[i, :] = axis_ratios
elif mask_type == 'shell':
    for i, (ri, ro) in enumerate(zip(radii[:-1], radii[1:])):
        #create the ID mask
        print('creating masks')
        com_mask = cmf.analysis.get_all_id_masks(snap, radius=10)

        #get com coordinates
        print('determining coms')
        xcom = cmf.analysis.get_coms_of_each_galaxy(snap, masks=com_mask)
        vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=com_mask)

        for ind, d in enumerate(xcom):
            star_id_masks = cmf.analysis.get_all_id_masks(snap, radius=[ri, ro])
            dm_id_masks = cmf.analysis.get_all_id_masks(snap, family='dm', radius=[ri, ro])
            if ind > 0 : continue
            print('BH: {}'.format(d))
            if family == 'stars':
                com_snap = snap.stars[star_id_masks[d]]
            elif family == 'dm':
                com_snap = snap.dm[dm_id_masks[d]]
            else:
                raise ValueError('must be stars or dm')
            particle_counts[i] = len(com_snap)
            com_snap['pos'] = com_snap['pos'] - xcom[d]
            com_snap['vel'] = com_snap['vel'] - vcom[d]
            #create the reduced inertia tensor
            print('determing I')
            inertia_tensor = pygad.analysis.reduced_inertia_tensor(com_snap)
            print('getting eigenvalues')
            eigen_vals, eigen_vecs = scipy.linalg.eig(inertia_tensor)
            idx = np.argsort(eigen_vals)[::-1]
            eigen_vals = np.real(eigen_vals[idx])
            eigen_vecs = eigen_vecs[:, idx]
            abc = np.sqrt(eigen_vals)
            print('axes')
            print(abc)
            axis_ratios = abc[1:] / abc[0]
            print(axis_ratios)
            ratios[i, :] = axis_ratios

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

fig, (ax1, ax2) = plt.subplots(1,2, sharex='all', sharey='all')
_,ax1,*_ = pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1, extent=12000)
for i in range(2):
    _,ax2,*_ = pygad.plotting.image(snap.dm[dm_id_masks[snap.bh['ID'][i]]], qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2, extent=12000)
plt.tight_layout()
plt.savefig('{}/{}_masks.png'.format(savedir, mask_type), dpi=300)