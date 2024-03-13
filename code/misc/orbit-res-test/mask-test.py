import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

#snapfile = '/Volumes/Rawlings_Storage/KETJU/data/merger/P01a_P01b/data/P01a_P01b_o95_000.hdf5'
snapfile = '/scratch/pjohanss/arawling/collisionless_merger/res-test/reduced-x-10/0-001/output/0001_000.hdf5'
snap = pygad.Snapshot(snapfile)
snap.to_physical_units()
idmasks = bgs.analysis.get_all_id_masks(snap)

plt.xlabel('Index position in Family')
plt.ylabel('Particle ID number')
stars = snap.stars[np.argsort(snap.stars['ID'])]
for i, bhid in enumerate(snap.bh['ID']):
    plt.plot(stars[idmasks[bhid]]['ID'], ls='', marker='.', label='stars_{}'.format(i))
#plt.plot(np.sort(snap.dm['ID']), ls='', marker='.', label='dm')
plt.plot(snap.bh['ID'], ls='', marker='.', label='bh')
#plt.plot(snap.bh['ID'][bhid], ls='', marker='x')
plt.legend()
#plt.savefig('./particle_ids.png', dpi=300)
plt.show()