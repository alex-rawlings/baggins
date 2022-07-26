import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad
from datetime import datetime
import psutil
import sys


if __name__ == "__main__":
    savefile = "/scratch/pjohanss/arawling/testing/iida_density.pickle"
    snapdir_ketju = "/scratch/pjohanss/ikostamo/test/ketju_run/output1"
    snapdir_gadget = "/scratch/pjohanss/ikostamo/test/ketju_run/output_no_ketju"
    r_edges = np.geomspace(5e-3, 300, 51)
    rng = np.random.default_rng(42)

    parser = argparse.ArgumentParser(description="Construct projected density estimates for time series data", allow_abbrev=False)
    parser.add_argument("-n", "--new", help="Run new analysis", action="store_true", dest="new")
    parser.add_argument("-o", "-obs", help="No. observations", type=int, dest="obs", default=3)
    args = parser.parse_args()

    n_workers = int(cmf.utils.get_cpu_count()/2)

    if args.new:
        data = {}
        print(f"RAM usage: {psutil.cpu_percent(4)}")

        x = cmf.mathematics.get_histogram_bin_centres(r_edges)

        for i, (snapdir, sim) in enumerate(zip((snapdir_ketju, snapdir_gadget),
                                                ("Ketju", "Gadget"))):
            data[sim] = {}
            data[sim]["x"] = x
            snapfiles = cmf.utils.get_snapshots_in_dir(snapdir)
            n = len(snapfiles)
            data[sim]["t"] = np.full(len(snapfiles), np.nan, dtype=float)
            data[sim]["density"] = {}
            for ii, snapfile in enumerate(snapfiles):
                cmf.LOGS.logger.info(f"{ii+1} of {n} snaps")
                snap = pygad.Snapshot(snapfile, physical=True)
                print(f"Snap size: {sys.getsizeof(snap)/(1024**2)} MB")
                data[sim]["t"][ii] = cmf.general.convert_gadget_time(snap)
                start_time = datetime.now()
                re, vsig, Sigma = cmf.analysis.projected_quantities(snap, obs=args.obs, r_edges=r_edges, rng=rng, n_workers=n_workers)
                end_time = datetime.now()
                print(f"Execution time: {end_time-start_time}")
                for j, (k,v) in enumerate(Sigma.items()):
                    if j > 0: break
                    data[sim]["density"][f"{data[sim]['t'][ii]:.1f}"] = v
                snap.delete_blocks()
                del snap
                print(f"RAM usage: {psutil.cpu_percent(4)}")
                pygad.gc_full_collect()
        cmf.utils.save_data(data, filename=savefile)
    else:
        data = cmf.utils.load_data(savefile)

    fig, ax = plt.subplots(1,2, sharex="all", sharey="all")
    ax[0].set_xlabel("r/kpc")
    ax[0].set_ylabel(r"$\rho$ / (M$_\odot$/kpc$^3$)")
    ax[1].set_xlabel("r/kpc")
    ax[0].set_xscale("log")
    ax[1].set_yscale("log")

    max_time = []
    for v in data.values():
        max_time.append(max(v["t"]))
    
    mpcol, sm = cmf.plotting.create_normed_colours(0, 1.1*max(max_time), cmap="plasma")

    for i, (k,v) in enumerate(data.items()):
        ax[i].set_title(k)
        for j, v2 in enumerate(v["density"].values()):
            if j<2: continue
            # TODO why is field "estimate" the same as "high"??
            #print(v2)
            #quit()
            ax[i].plot(v["x"], np.nanmedian(v2, axis=0), c=mpcol(v["t"][j]))
            ax[i].fill_between(v["x"], y1=v2["low"], y2=v2["high"], alpha=0.4, fc=mpcol(v["t"][j]))
    cbar = plt.colorbar(sm, ax=ax[1])
    cbar.ax.set_ylabel("t/Gyr")
    cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "other_tests/iida.png"))
    if not args.new: plt.show()