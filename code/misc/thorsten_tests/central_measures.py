import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad

data_dirs = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05"
]

r_edges = [0, 0.3]
data = {}

SL = cmf.CustomLogger("script_logger", console_level="INFO")

for d in data_dirs:
    merger_class = d.rstrip("/").split("/")[-1]
    for i in range(10):
        k = f"{merger_class}-{i:03d}"
        print(f"Reading: {k}", flush=True)
        data[k] = {}
        pdir = os.path.join(d, f"perturbations/{i:03d}/output")
        for j, s in enumerate(cmf.utils.get_snapshots_in_dir(pdir)):
            snap = pygad.Snapshot(s, physical=True)
            t = cmf.general.convert_gadget_time(snap, "Myr")
            kk = f"{t:.2f}"
            data[k][kk] = {}

            xcom = cmf.analysis.get_com_of_each_galaxy(snap, method="ss", family="stars")
            vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom)
            SL.logger.debug(xcom)
            bh_id = cmf.analysis.get_massive_bh_ID(snap.bh)
            beta, bc = cmf.analysis.velocity_anisotropy(snap, r_edges=r_edges, xcom=xcom[bh_id], vcom=vcom[bh_id])
            
            _Re, _vsig2Re, _vsig2r, _Sigma = cmf.analysis.projected_quantities(snap, obs=2, r_edges=r_edges)
            Sigma = np.nanmedian(list(_Sigma.values())[0], axis=0)

            data[k][kk]["Sigma"] = Sigma
            data[k][kk]["beta"] = beta

            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap
    # save data each child so we don't lose everything
    cmf.utils.save_data(data, os.path.join(cmf.DATADIR, f"mergers/processed_data/thorsten_centrals.pickle"))


