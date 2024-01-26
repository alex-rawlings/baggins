import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


snapfiles = cmf.utils.read_parameters(os.path.join(
                                            cmf.HOME,
                                            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml"
))

snapshots = dict()
for k, v in snapfiles["snap_nums"].items():
    if k == "v1020": break
    snapshots[k] = os.path.join(snapfiles["parent_dir"], f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5")

for i, (k, v) in enumerate(snapshots.items()):
    if i!=10:continue
    print(f"Creating IFU maps for {k}")
    #v = "/scratch/pjohanss/arawling/antti-nugget/rotators/fast/fastrotator.hdf5"
    snap = pygad.Snapshot(v, physical=True)
    pygad.analysis.orientate_at(snap, mode="red I", total=True)
    ball_mask = pygad.BallMask(
        R = 0.5*pygad.analysis.half_mass_radius(snap.stars),
        center = pygad.analysis.shrinking_sphere(
            snap.stars,
            pygad.analysis.center_of_mass(snap.stars),
            30),
    )


    voronoi_stats = cmf.analysis.voronoi_binned_los_V_statistics(
        snap.stars[ball_mask]["pos"][:,0],
        snap.stars[ball_mask]["pos"][:,1],
        snap.stars[ball_mask]["vel"][:,2],
        snap.stars[ball_mask]["mass"], 
        100,
        part_per_bin=1000
    )
    #cmf.plotting.voronoi_plot(voronoi_stats)

    rs, h4r = cmf.analysis.radial_profile_velocity_moment(voronoi_stats, "h4")
    plt.plot(rs, h4r, marker="o")
plt.show()