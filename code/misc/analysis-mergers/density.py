import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
from datetime import datetime


if __name__ == "__main__":
    iida = True
    if iida:
        savefile = "/scratch/pjohanss/arawling/testing/iida_density.pickle"
        snapdirs = dict(
                        ketju = "/scratch/pjohanss/ikostamo/test/ketju_run/output1",
                        gadget = "/scratch/pjohanss/ikostamo/test/ketju_run/output_no_ketju",
                        no_bh = "/scratch/pjohanss/ikostamo/test/ketju_no_bh/output",
                        gadget_starExtraSoft = "/scratch/pjohanss/ikostamo/test/ketju_run/output_no_ketju_starsoftening/"
        )
        figname = "iida_2d"
    else:
        savefile = "/scratch/pjohanss/arawling/testing/alex_density.pickle"
        snapdirs = dict(
                        run2 = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/002/output"
        )
        figname = "alex"
    r_edges = np.geomspace(1e-2, 100, 51)
    rng = np.random.default_rng(42)
    logs = bgs.ScriptLogger("density_logs", "INFO")

    parser = argparse.ArgumentParser(description="Construct projected density estimates for time series data", allow_abbrev=False)
    parser.add_argument("-n", "--new", help="Run new analysis", action="store_true", dest="new")
    parser.add_argument("-o", "-obs", help="No. observations", type=int, dest="obs", default=3)
    args = parser.parse_args()

    if args.new:
        data = {}

        x = bgs.mathematics.get_histogram_bin_centres(r_edges)

        for i, (sim, snapdir) in enumerate(snapdirs.items()):
            data[sim] = {}
            data[sim]["x"] = x
            snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
            n = len(snapfiles)
            data[sim]["t"] = np.full(len(snapfiles), np.nan, dtype=float)
            data[sim]["density"] = {}
            data[sim]["vsig_r"] = {}
            for ii, snapfile in enumerate(snapfiles):
                logs.logger.info(f"{ii+1} of {n} snaps")
                print(f"Reading {snapfile}")
                snap = pygad.Snapshot(snapfile, physical=True)
                data[sim]["t"][ii] = bgs.general.convert_gadget_time(snap)
                #idmask = bgs.analysis.get_all_id_masks(snap)
                start_time = datetime.now()
                re, vsig, vsig_r, Sigma = bgs.analysis.projected_quantities(snap, obs=args.obs, r_edges=r_edges, rng=rng)
                end_time = datetime.now()
                print(f"Execution time: {end_time-start_time}")
                _kk = f"{data[sim]['t'][ii]:.1f}"
                for j, (k,v) in enumerate(Sigma.items()):
                    if j > 0: break
                    data[sim]["density"][_kk] = v
                for j, (k,v) in enumerate(vsig_r.items()):
                    if j>0: break
                    data[sim]["vsig_r"][_kk] = v
                snap.delete_blocks()
                del snap
                pygad.gc_full_collect()
        bgs.utils.save_data(data, filename=savefile)
    else:
        data = bgs.utils.load_data(savefile)

    # set up figure
    fig, ax = plt.subplots(2,len(data), sharex="all", sharey="row", squeeze=False)
    ax[0,0].set_ylabel(r"$\Sigma(r)$ / (M$_\odot$/kpc$^2$)")
    ax[1,0].set_ylabel(r"$\sigma_\star(r)$ / (km/s)")
    ax[0,0].set_xscale("log")
    ax[0,0].set_yscale("log")
    for i in range(len(data)):
        ax[-1,i].set_xlabel(r"$r$/kpc")

    max_time = []
    for v in data.values():
        max_time.append(max(v["t"]))
    
    mpcol, sm = bgs.plotting.create_normed_colours(0, 1.1*max(max_time), cmap="plasma")

    for i, (k,v) in enumerate(data.items()):
        ax[0,i].set_title(k)
        for j, (dens_v, sig_v) in enumerate(zip(v["density"].values(), v["vsig_r"].values())):
            if v["t"][j] < 0.25: continue
            if not iida and j>len(v["density"])-2: continue
            # plot density
            ax[0,i].plot(v["x"], np.nanmedian(dens_v, axis=0), c=mpcol(v["t"][j]))
            ax[0,i].fill_between(v["x"], y1=np.nanquantile(dens_v, 0.75, axis=0), y2=np.nanquantile(dens_v, 0.25, axis=0), alpha=0.4, fc=mpcol(v["t"][j]))
            # plot velocity dispersion
            ax[1,i].plot(v["x"], np.sqrt(np.nanmedian(sig_v, axis=0)), c=mpcol(v["t"][j]))
            ax[1,i].fill_between(v["x"], y1=np.sqrt(np.nanquantile(sig_v, 0.75, axis=0)), y2=np.sqrt(np.nanquantile(sig_v, 0.25, axis=0)), alpha=0.4, fc=mpcol(v["t"][j]))
    cbar = plt.colorbar(sm, ax=ax[:,-1])
    cbar.ax.set_ylabel("t/Gyr")
    bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"other_tests/{figname}.png"))
    if not args.new: plt.show()
