import pygad
import cm_functions as cmf

pidx = "002"
snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/A-D-3.0-1.0/perturbations/{pidx}/output/AD_perturb_{pidx}_000.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

id_masks = cmf.analysis.get_all_id_masks(snap)

for m in ("pot", "ss"):
    xcoms = cmf.analysis.get_com_of_each_galaxy(snap, method=m, masks=id_masks)
    print(xcoms)