import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
import multiprocessing as mp
from functools import partial


def extract_helper(i, data):
    k = f"{merger_class}-{i:03d}"
    data[k] = {}
    # need to create copy so that global variable is updated
    temp_data = data[k]
    pdir = os.path.join(d, f"perturbations/{i:03d}/output")
    for j, s in enumerate(bgs.utils.get_snapshots_in_dir(pdir)):
        snap = pygad.Snapshot(s, physical=True)
        t = bgs.general.convert_gadget_time(snap, "Myr")
        kk = f"{t:.2f}"
        temp_data[kk] = {}

        xcom = bgs.analysis.get_com_of_each_galaxy(snap, method="ss", family="stars")
        vcom = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcom)
        SL.debug(xcom)
        bh_id = bgs.analysis.get_massive_bh_ID(snap.bh)
        beta, bc = bgs.analysis.velocity_anisotropy(snap, r_edges=r_edges, xcom=xcom[bh_id], vcom=vcom[bh_id])
        
        _Re, _vsig2Re, _vsig2r, _Sigma = bgs.analysis.projected_quantities(snap, obs=2, r_edges=r_edges)
        Sigma = np.nanmedian(list(_Sigma.values())[0], axis=0)

        temp_data[kk]["Sigma"] = Sigma
        temp_data[kk]["beta"] = beta

        snap.delete_blocks()
        pygad.gc_full_collect()
        del snap
    # reassign copy back to global variable
    data[k] = temp_data
    print(f"Completed: {k}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot central (<300pc) density, beta", allow_abbrev=False)
    parser.add_argument("-e", "--extract", help="extract new data", dest="extract", action="store_true")
    args = parser.parse_args()

    save_name = os.path.join(bgs.DATADIR, f"mergers/processed_data/thorsten_centrals.pickle")
    if args.extract:
        data_dirs = [
            "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05"
        ]

        r_edges = [0, 0.3]
        manager = mp.Manager()
        data = manager.dict()
        extract_helper_P = partial(extract_helper, data=data)

        SL = bgs.setup_logger("script_logger", console_level="INFO")
        for d in data_dirs:
            merger_class = d.rstrip("/").split("/")[-1]

            with mp.Pool(10) as pool:
                pool.map(extract_helper_P, [i for i in range(10)])
        bgs.utils.save_data(data.copy(), save_name)
    else:
        data = bgs.utils.load_data(save_name)
        hmq_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.05"
    
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel(r"$\beta(r)$")
        ax.set_ylabel(r"$\Sigma(r)/\mathrm{M}_\odot\mathrm{kpc}^{-2}$")
        cnames = list(data.keys())
        cnames.sort()
        for (cname, hmqfile) in zip(cnames, bgs.utils.get_files_in_dir(hmq_dir)):
            hmq = bgs.analysis.HMQuantitiesBinaryData.load_from_file(hmqfile)
            child_data = list(data[cname].values())[-2]
            ax.scatter(child_data["beta"], child_data["Sigma"], label=cname, alpha=(1 if hmq.merger_remnant["merged"] else 0.3), linewidth=0.3, ec="k")
        ax.legend(fontsize="x-small")
        bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "other_tests/thorsten/AC-3.0-0.05-centrals.png"), fig)
        plt.show()




