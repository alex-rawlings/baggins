import numpy as np
from scipy.spatial.transform import Rotation
import pygad


snapfiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_002.hdf5",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_073.hdf5"
]

for snapfile in snapfiles:
    snap = pygad.Snapshot(snapfile, physical=True)
    #CENTERING
    star_center=np.mean(snap.stars["pos"], axis=0)
    star_center=pygad.analysis.shrinking_sphere(snap.stars, star_center, 300.)  #shrinking sphere technique to find the point with the highest stellar density
    star_velcenter=np.mean(snap.stars[np.linalg.norm(snap.stars['pos']-star_center, axis=1)<1.]['vel'], axis=0)  #velocity of central kpc
    translation = pygad.Translation(-star_center)
    translation.apply(snap, total=True)
    boost = pygad.Boost(-star_velcenter)
    boost.apply(snap, total=True)
    if len(snap.bh)>0:  #if the simulation has black holes within the cut radius...
        bhball=snap.bh[pygad.BallMask(10)]
        if len(bhball) > 0:
            #  BHs in search radius
            bh_center_of_mass=np.average(bhball["pos"], weights=bhball['mass'], axis=0) # calculates center of mass of the black holes
            bh_velcenter=np.average(bhball["vel"], weights=bhball['mass'], axis=0)  #velocity of center of mass
            translation = pygad.Translation(-bh_center_of_mass)
            translation.apply(snap, total=True)
            boost = pygad.Boost(-bh_velcenter)
            boost.apply(snap,total=True)
    #ORIENTING  (according to the reduced inertia tensor of the 50% most bound star particles)
    try:
        binding_energy=snap.stars['E']  #total (binding) energy
        en_sort_index=np.argsort(binding_energy) #index that orders particles according to their binding energy
        en_threshold=binding_energy[en_sort_index][int(0.5*len(binding_energy))]  #energy threshold
        orient_s=snap.stars[binding_energy<en_threshold] #sub-snapshot that includes only the 50% most bound particles
    except (ValueError, IndexError):
        print("No potential information in the snapshot: using whole snapshot for orientation")
        orient_s = snap.stars
    redI = pygad.analysis.reduced_inertia_tensor(orient_s)
    # get eigenvalues and eigenvectors of reduced inertia tensor
    vals, vecs = np.linalg.eigh(redI)
    i = np.argsort(vals)[::-1]
    try:
        T = pygad.Rotation(vecs[:, i].T)
    except ValueError:
        # probably not a proper rotation... (not right-handed)
        vecs[:, i[1]] *= -1
        T = pygad.Rotation(vecs[:, i].T)
    except:
        raise
    print(f"Rotated {T.angle()*180/np.pi:.2f} degrees about {T.axis()}")
    print(T.rotmat)
    r = Rotation.from_matrix(T.rotmat)
    print(f"Euler angles: {r.as_euler('xyz', degrees=True)}")