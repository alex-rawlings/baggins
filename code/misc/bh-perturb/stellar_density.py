import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad 
import cm_functions as cmf


main_path = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc"
subpath = "output/"

fig, ax = plt.subplots(1,1)

ball_radius = pygad.UnitScalar(0.1, "kpc")

for i, galname in enumerate(("NGCa0524", "NGCa2986", "NGCa3348", "NGCa3607", "NGCa4291")):
    snapdir = os.path.join(main_path, galname, subpath)
    snap_list = cmf.utils.get_snapshots_in_dir(snapdir)
    stellar_density = np.full_like(snap_list, np.nan, dtype=float)
    times = np.full_like(stellar_density, np.nan)
    for j, snapfile in enumerate(snap_list):
        print("Reading {:.1f}%                              ".format(j/(len(snap_list)-1)*100), end="\r")
        snap = pygad.Snapshot(snapfile, physical=True)
        times[j] = cmf.general.convert_gadget_time(snap)
        xcom = pygad.analysis.shrinking_sphere(snap, center=snap.bh["pos"], R=10)
        ball_mask = pygad.BallMask(ball_radius, center=xcom)
        stellar_ball_mass = len(snap.stars[ball_mask])*snap.stars["mass"][0]
        stellar_density[j] = stellar_ball_mass / (4*np.pi/3*ball_radius**3)
        del snap
    print("Complete                              ")
    ax.plot(times, stellar_density, label=galname)
ax.legend()
ax.set_xlabel("t/Gyr")
ax.set_ylabel(r"$\rho$ / (M$_\odot$/kpc$^3$)")
plt.show()