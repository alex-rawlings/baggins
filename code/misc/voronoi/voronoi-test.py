import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


datadir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.005/perturbations/000/output/"
snapfile = cmf.utils.get_snapshots_in_dir(datadir)[-1]
snap = pygad.Snapshot(snapfile, physical=True)
pygad.analysis.orientate_at(snap, mode="L")
#r_half_mass_proj = list(cmf.analysis.projected_half_mass_radius(snap).values())[0]

inner_mask = pygad.BallMask(pygad.UnitQty(10, "kpc"), center=pygad.analysis.center_of_mass(snap.bh))


voronoi_stats = cmf.analysis.voronoi_binned_los_V_statistics(
    snap.stars[inner_mask]["pos"][:,0],
    snap.stars[inner_mask]["pos"][:,1],
    snap.stars[inner_mask]["vel"][:,2],
    snap.stars[inner_mask]["mass"], 
    300,
    part_per_bin=1000
)

cmf.plotting.voronoi_plot(voronoi_stats)
plt.show()