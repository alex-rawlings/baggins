import numpy as np
import pygad
from scipy.interpolate import griddata
import pickle
import gc


X,Y = np.meshgrid(*2*[np.linspace(-2,2,400)])
pots = []
times = []
for i in range(30,50):
    snap = pygad.Snapshot(f'output_htr/snap_{i:03d}.hdf5'); snap.to_physical_units()

    L = np.cross(np.diff(snap.bh['pos'],axis=0),np.diff(snap.bh['vel'],axis=0))
    Lhat = L/np.linalg.norm(L)
    stars = snap.stars[abs(np.dot(snap.stars['pos'],Lhat[0]))<1e-2]
    pot = np.array(griddata(stars['pos'][:,[0,2]], stars['Epot'], (X,Y)))
    pots.append(pot)
    times.append(float(pygad.UnitQty(snap.time, 'kpc/(km/s)').in_units_of('Gyr')))

    snap.delete_blocks()
    gc.collect()

with open('hernquist_merger_potential.pkl', 'wb') as f:
    pickle.dump(dict(X=X, Y=Y, pots=pots, times=times,),
                f, protocol=-1)

