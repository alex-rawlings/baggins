import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


def masked_output(x, qcut=0.99):
    mask = np.abs(x) < np.quantile(np.abs(x), qcut)
    return x[mask]

snapfiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output/A05-C05-3.0-0.001_014.hdf5", 
    #"/scratch/pjohanss/arawling/collisionless_merger/stability-tests/triaxial/NGCa0524t/output/NGCa0524t_059_aligned.hdf5"
    "/scratch/pjohanss/arawling/collisionless_merger/regen-test/recentred-new/high-softening/ACH.hdf5",
    #"/scratch/pjohanss/arawling/collisionless_merger/regen-test/recentred/high-softening/ACH.hdf5"
]

fig, ax = plt.subplots(3,4, sharex="col")

for snapfile in snapfiles:
    print("Reading: {}".format(snapfile))
    snap = pygad.Snapshot(snapfile, physical=True)
    star_id_masks = cmf.analysis.get_all_id_masks(snap, "stars")
    dm_id_masks = cmf.analysis.get_all_id_masks(snap, "dm")
    star_xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks, family="stars")
    dm_xcom = cmf.analysis.get_com_of_each_galaxy(snap, 100, masks=dm_id_masks, family="dm")
    star_radial_mask = cmf.analysis.get_all_radial_masks(snap, 10000, centre=star_xcom, id_masks=star_id_masks, family="stars")
    dm_radial_mask = cmf.analysis.get_all_radial_masks(snap, 10000, centre=dm_xcom, id_masks=dm_id_masks, family="dm")
    for i in range(3):
        #ax[i,0].hist(masked_output(snap.stars["pos"][:,i]), 100, density=True, alpha=0.3)
        #ax[i,1].hist(masked_output(snap.dm["pos"][:,i]), 100, density=True, alpha=0.3)
        #ax[i,2].hist(masked_output(snap.stars["vel"][:,i]), 100, density=True, alpha=0.3)
        #ax[i,3].hist(masked_output(snap.dm["vel"][:,i]), 100, density=True, alpha=0.3)
        for key in star_xcom.keys():
            ax[i,0].hist(masked_output(snap.stars[star_radial_mask[key]]["pos"][:,i]), 100, density=True, alpha=0.3)
            ax[i,1].hist(snap.dm[dm_radial_mask[key]]["pos"][:,i], 100, density=True, alpha=0.3)
            ax[i,2].hist(snap.stars[star_radial_mask[key]]["vel"][:,i], 100, density=True, alpha=0.3)
            ax[i,3].hist(snap.dm[dm_radial_mask[key]]["vel"][:,i], 100, density=True, alpha=0.3)
ax[0,0].set_title("Star: Position")
ax[0,1].set_title("DM: Position")
ax[0,2].set_title("Star: Velocity")
ax[0,3].set_title("DM: Velocity")
plt.show()