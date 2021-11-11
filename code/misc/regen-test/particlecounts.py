import pygad

snapA = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/triaxial/NGCa0524t/output/NGCa0524t_094.hdf5"
snapC = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/triaxial/NGCa3348t/output/NGCa3348t_097.hdf5"
snapAC = "/scratch/pjohanss/arawling/collisionless_merger/regen-test/recentred/high-softening/output/ACH-HS_012.hdf5"

pcounts = dict(
    stars = 0,
    dm = 0,
    bh = 0
)

for i, snapfile in enumerate((snapA, snapC, snapAC)):
    snap = pygad.Snapshot(snapfile)
    snap.to_physical_units()
    if snapfile == snapAC:
        pcounts = dict.fromkeys(pcounts, 0)
    for fam in ("stars", "dm", "bh"):
        subsnap = getattr(snap, fam)
        pcounts[fam] += len(subsnap["ID"])
    if i > 0:
        for k,v in pcounts.items():
            print("{}: {}".format(k,v))
    print("-----")