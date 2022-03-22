import pygad

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/000/output/AC_perturb_000_000.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

print(snap.cosmology)