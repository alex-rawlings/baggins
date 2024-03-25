import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


class COMobject:
    def __init__(self, n, k1, k2):
        self.xcom = {k1:np.full((n, 3), np.nan), k2:np.full((n, 3), np.nan)}
        self.vcom = {k1:np.full((n, 3), np.nan), k2:np.full((n, 3), np.nan)}


snapdir = "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output"
extract_idx = 14
brownian = {"pos":0.012, "vel":14}

snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
snaptimes = np.full_like(snapfiles, np.nan, dtype=float)

for ind, snapfile in enumerate(snapfiles):
    print("{}: {}".format(ind, snapfile))
    snap = pygad.Snapshot(snapfile)
    snap.to_physical_units()
    snaptimes[ind] = bgs.general.convert_gadget_time(snap, new_unit="Myr")
    if ind == 0:
        id_masks = dict(
            stars = bgs.analysis.get_all_id_masks(snap),
            dm = bgs.analysis.get_all_id_masks(snap, family="dm")
        )
        com = dict(
            stars = COMobject(len(snapfiles), *list(id_masks["stars"].keys())),
            dm = COMobject(len(snapfiles), *list(id_masks["dm"].keys()))
        )
    for ind2, (family, r0) in enumerate(zip(("stars", "dm"), (100, 1000))):
        xcom = bgs.analysis.get_com_of_each_galaxy(snap, masks=id_masks[family], family=family, initial_radius=r0)
        vcom = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=id_masks[family], family=family)
        for ind3, bhid in enumerate(id_masks["stars"].keys()):
            com[family].xcom[bhid][ind, :] = xcom[bhid]
            com[family].vcom[bhid][ind, :] = vcom[bhid]
for bhid in com["dm"].vcom.keys():
    print(com["stars"].vcom[bhid][:,2])
    print(com["dm"].vcom[bhid][:,2])
    print("---------")

fig, ax = plt.subplots(3,2, sharex="all", sharey="col")
label_coords = ["x", "y", "z"]
for i in range(3):
    ax[i,0].set_ylabel(r"$\Delta${}/kpc".format(label_coords[i]))
    ax[i,1].set_ylabel(r"$\Delta$v$_{}$/km/s".format(label_coords[i]))
    ax[i,0].axvline(snaptimes[extract_idx], c="tab:red", lw=0.8)
    ax[i,1].axvline(snaptimes[extract_idx], c="tab:red", lw=0.8)
    ax[i,0].fill_between(snaptimes, brownian["pos"], -brownian["pos"], alpha=0.3, color="k")
    ax[i,1].fill_between(snaptimes, brownian["vel"], -brownian["vel"], alpha=0.3, color="k")
for i in range(2):
    ax[-1,i].set_xlabel("t/Myr")

for i in range(3):
    for j, bhid in enumerate(com["stars"].xcom.keys()):
        ax[i,0].plot(snaptimes, com["stars"].xcom[bhid][:,i] - com["dm"].xcom[bhid][:,i], label=("{}".format(bhid) if i==0 else ""))
        ax[i,1].plot(snaptimes, com["stars"].vcom[bhid][:,i] - com["dm"].vcom[bhid][:,i])
ax[0,0].legend()
plt.show()
