import os
import numpy as np
import matplotlib.pyplot as plt
import pygad
import gadgetorbits as go
import baggins as bgs

bgs.plotting.check_backend()

base_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/"

snapdirs = [d.path for d in os.scandir(base_dir) if d.is_dir() and "kick-" in d.name]
snapdirs.sort()

mergemask = [
    6,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    0,
    5,
    6,
    0,
    0,
    4,
]
minrad = 0.2
maxrad = 30
nbin = 10

fig, ax = plt.subplots()
ax.set_xlabel("Kick velocity [km/s]")
ax.set_ylabel("Energy")

for i, snapdir in enumerate(snapdirs):
    if i%4 != 0: continue
    '''snapfile = os.path.join(snapdir, "output/snap_016.hdf5")
    print(f"Reading: {snapfile}")
    snap = pygad.Snapshot(snapfile, physical=True)
    IDs, frac, bh_energy = bgs.analysis.find_individual_bound_particles(snap, return_extra=True)

    print(bh_energy)'''

    kickvel = float(os.path.basename(snapdir).replace("kick-vel-", ""))

    '''ax.plot(kickvel, bh_energy, ls="", marker="o", c="tab:blue")

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()'''

    # read in orbit analysis classification
    orbitfile = os.path.join(base_dir, f"orbit_analysis/{os.path.basename(snapdir)}/classification/{os.path.basename(snapdir)}.cl")
    print(orbitfile)
    (
        orbitids,
        classids,
        rad,
        rot_dir,
        energy,
        denergy,
        inttime,
        b92class,
        pericenter,
        apocenter,
        meanposrad,
        minangmom,
    ) = go.loadorbits(orbitfile, mergemask=mergemask, addextrainfo=True)

    mask = np.logical_and(rad<1, classids==4)
    energy_mean = np.nanmean(energy[mask])
    energy_sd = np.nanstd(energy[mask])

    ax.errorbar(kickvel, energy_mean, yerr=energy_sd, ls="", marker="o", c="tab:orange")


plt.savefig("energies.png", dpi=300)

plt.show()
