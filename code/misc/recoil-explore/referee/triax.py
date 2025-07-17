import pygad
import baggins as bgs


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_002.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)

rhalf = pygad.analysis.half_mass_radius(snap.stars)
rvir,_ = pygad.analysis.virial_info(snap)


for r in (rhalf, rvir):
    mask = pygad.BallMask(r)

    ba, ca = bgs.analysis.get_galaxy_axis_ratios(snap[mask], family="stars")

    print(mask.R)
    print(f"b/a: {ba:.3f}")
    print(f"c/a: {ca:.3f}")