import pygad
import baggins as bgs

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_004.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

xcom = pygad.Translation(
    -pygad.analysis.shrinking_sphere(
        snap.stars,
        pygad.analysis.center_of_mass(snap.stars),
        30
    )
)
xcom.apply(snap, total=True)

rinfl = list(bgs.analysis.influence_radius(snap).values())[0]

print(f"Influence radius is {rinfl:.2e}")
