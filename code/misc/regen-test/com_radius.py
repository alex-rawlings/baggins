import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


snapfiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output/A05-C05-3.0-0.001_014.hdf5"
    #"/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/A05-C05-3.0-0.001.hdf5"
]

fig, ax = plt.subplots(3,4, sharex="col", sharey="col")
cols = bgs.plotting.mplColours()
rshells_stars = np.geomspace(1e-6, 500, 20)
rshells_dm = np.geomspace(1e-6, 5000, 20)
for snapfile in snapfiles:
    snap = pygad.Snapshot(snapfile, physical=True)
    #mask galaxies by id
    star_id_masks = bgs.analysis.get_all_id_masks(snap, "stars")
    dm_id_masks = bgs.analysis.get_all_id_masks(snap, "dm")
    #determine the reference com
    star_xcom = bgs.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks, family="stars")
    dm_xcom = bgs.analysis.get_com_of_each_galaxy(snap, 100, masks=dm_id_masks, family="dm")
    for (riS, roS, riH, roH) in zip(
        rshells_stars[:-1], rshells_stars[1:], 
        rshells_dm[:-1], rshells_dm[1:]
    ):
        #create shell masks
        star_radial_masks = bgs.analysis.get_all_radial_masks(snap, (riS, roS), centre=star_xcom, id_masks=star_id_masks, family="stars")
        dm_radial_masks = bgs.analysis.get_all_radial_masks(snap, (riH, roH), centre=dm_xcom, id_masks=dm_id_masks, family="dm")
        #recompute com motions for particles in this shell
        for i, bhid in enumerate(star_id_masks.keys()):
            for j, (family, radial_masks, radius) in enumerate(zip(("stars", "dm"), (star_radial_masks, dm_radial_masks), (riS, riH))):
                subsnap = getattr(snap, family)
                xcom = pygad.analysis.mass_weighted_mean(subsnap[radial_masks[bhid]], qty="pos")
                vcom = pygad.analysis.mass_weighted_mean(subsnap[radial_masks[bhid]], qty="vel")
                for k in range(3):
                    ax[k, j].scatter(radius, xcom[k], c=cols[i], marker=".")
                    ax[k, j+2].scatter(radius, vcom[k], c=cols[i], marker=".")
for axi in ax[-1,:]:
    axi.set_xlabel("r/kpc")
    axi.set_xscale("log")
ax[0,0].set_title("Stars: xcom")
ax[0,1].set_title("DM: xcom")
ax[0,2].set_title("Stars: vcom")
ax[0,3].set_title("DM: vcom")
labvals = ["x", "y", "z"]
for i, lv in enumerate(labvals):
    ax[i,0].set_ylabel("{}".format(lv))
    ax[i,1].set_ylabel("{}".format(lv))
    ax[i,2].set_ylabel(r"v$_{}$".format(lv))
    ax[i,3].set_ylabel(r"v$_{}$".format(lv))

plt.show()