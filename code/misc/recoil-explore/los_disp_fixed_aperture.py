# get the LOS velocity dispersion for stars in a 200pc aperture about the cluster
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs

bgs.plotting.check_backend()

snapfiles = bgs.utils.get_snapshots_in_dir(f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")[:31]

times = np.full_like(snapfiles, np.nan, dtype=float)
sigma = np.full_like(times, np.nan)

for i, snapfile in enumerate(snapfiles):
    print(f"Doing snapshot {i:03d}")
    snap = pygad.Snapshot(snapfile, physical=True)
    mask = pygad.BallMask("200 pc", center=pygad.analysis.center_of_mass(snap.bh))
    times[i] = bgs.general.convert_gadget_time(snap)
    sigma[i] = pygad.analysis.los_velocity_dispersion(snap.stars[mask], proj=1)

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

fig, ax = plt.subplots()
ax.plot(times, sigma)
ax.set_xlabel("t/Gyr")
ax.set_ylabel("LOS dispersion in 200pc aperture [km/s]")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kick-survey", "200pc_aperture_sigma.png"))
