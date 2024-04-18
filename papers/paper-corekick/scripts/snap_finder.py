import os
import baggins as bgs


mainpath = (
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
)
target_time = 0.075

child_dirs = {}
for d in os.scandir(mainpath):
    if "kick-vel" in d.name:
        child_dirs[d.name] = os.path.join(d.path, "output")

for k in sorted(child_dirs):
    snaplist = bgs.utils.get_snapshots_in_dir(child_dirs[k])
    snap_num = bgs.analysis.snap_num_for_time(snaplist, target_time, "Gyr")
    print(f"{k}: {snap_num:03d}")
