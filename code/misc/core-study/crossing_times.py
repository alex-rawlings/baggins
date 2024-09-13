import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


snapfile = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick")[0]

snap = pygad.Snapshot(snapfile, physical=True)

xcom = bgs.analysis.get_com_of_each_galaxy(snap, method="ss", family="stars")
pygad.Translation(-list(xcom.values())[0]).apply(snap, total=True)

secs_per_Gyr = 3.154e7 * 1e9
vkicks = pygad.UnitArr(np.arange(0, 1021, 60), units="km/s")

for core_rad_val, lab in zip((0.414, 0.795, 0.58), ("low", "high", "med")):
    core_radius = pygad.UnitScalar(core_rad_val, units="kpc") # convert from kpc to km


    ball_mask = pygad.BallMask(core_radius)
    vel_disp = np.nanstd(snap.stars[ball_mask]["vel"])
    print(f"Velocity dispersion is {vel_disp:.2e} km/s")
    tcross = pygad.UnitScalar(core_radius/vel_disp, units="Gyr")

    print(f"Core crossing time ({lab}) is {tcross:.2e} Gyr")

    kick_time = pygad.UnitArr(core_radius/vkicks, units="Gyr")

    equiv_vel = np.interp(tcross, kick_time.view(np.ndarray)[::-1], vkicks.view(np.ndarray)[::-1])
    print(f"  --> which is {equiv_vel:.3e} km/s")

plt.semilogy(vkicks, kick_time, lw=2, label="Kick timescale")
plt.axhline(tcross, label="Core crossing", c="gray", lw=2)
plt.xlabel("vkick [km/s]")
plt.ylabel("Timescale [Gyr]")
plt.legend()

plt.show()