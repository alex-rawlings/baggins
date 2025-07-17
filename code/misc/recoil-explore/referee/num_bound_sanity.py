import pygad
import baggins as bgs


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-1800/output/snap_025.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)

print(f"BH distance: {snap.bh['r']}")

strong_bound_ids = bgs.analysis.find_strongly_bound_particles(snap)
N = len(strong_bound_ids)
print(f"There are {N} strongly bound stars")
print(f"This is {snap.stars['mass'][0] * N:.2e}")