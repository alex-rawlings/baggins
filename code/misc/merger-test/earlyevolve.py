import argparse
import os.path
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad

def find_nearest_star(snap, bhid):
    bhmask = pygad.IDMask(bhid)
    ballmask = pygad.BallMask(pygad.UnitScalar(2, units="kpc"), center=snap[bhmask]["pos"])
    r = pygad.utils.geo.dist(snap.stars[ballmask]["pos"], snap[bhmask]["pos"].flatten())
    return np.nanmin(r)

def plotter(ax, data, label, plot_median=True):
    alpha = (0.4 if plot_median else 0.9)
    times = data["times"]
    infl_rad = data["infl_rad"]
    nearest_star = data["nearest_star"]
    l = ax[0,0].plot(times, infl_rad["bh1"], alpha=alpha)
    ax[0,1].plot(times, infl_rad["bh2"], label=label, alpha=alpha)
    ax[1,0].plot(times, nearest_star["bh1"], alpha=alpha)
    ax[1,1].plot(times, nearest_star["bh2"], alpha=alpha)
    median_kwargs = {"alpha":0.9, "c":l[-1].get_color(), "lw":3}
    if plot_median:
        ax[0,0].axhline(np.nanmedian(infl_rad["bh1"]), **median_kwargs)
        ax[0,1].axhline(np.nanmedian(infl_rad["bh2"]), **median_kwargs)
        ax[1,0].axhline(np.nanmedian(nearest_star["bh1"]), **median_kwargs)
        ax[1,1].axhline(np.nanmedian(nearest_star["bh2"]), **median_kwargs)


def _mphelper(perturbNum, ax):
    times = []
    infl_rad = {"bh1":[], "bh2":[]}
    nearest_star = {"bh1":[], "bh2":[]}
    snapfiles = bgs.utils.get_snapshots_in_dir(os.path.join(mainpath, perturbNum, "output"))
    for snapfile in snapfiles:
        snap = pygad.Snapshot(snapfile, physical=True)
        bhids = snap.bh["ID"]
        bhids.sort()
        times.append(bgs.general.convert_gadget_time(snap, new_unit="Myr"))
        rh = bgs.analysis.influence_radius(snap, binary=False)
        infl_rad["bh1"].append(rh[bhids[0]])
        infl_rad["bh2"].append(rh[bhids[1]])
        nearest_star["bh1"].append(find_nearest_star(snap, bhids[0]))
        nearest_star["bh2"].append(find_nearest_star(snap, bhids[1]))
    data = dict(
        times = times,
        infl_rad = infl_rad,
        nearest_star = nearest_star
    )
    bgs.utils.save_data(data, f"./pickle/early-{perturbNum}.pickle")
    # plot
    plotter(ax, data, perturbNum)



mainpath = "/scratch/pjohanss/arawling/collisionless_merger/high-time-output/A-C-3.0-0.05-H"


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser(description="Plot key features of early evolution of a family of simulations", allow_abbrev=False)
    parser.add_argument("-n", "--new", help="Analyse new dataset?", dest="new", action="store_true")
    args = parser.parse_args()
    
    num_subdirs = 10

    fig, ax = plt.subplots(2,2, sharex="all", sharey="row")
    ax[0,0].set_yscale("log")
    ax[1,0].set_yscale("log")
    ax[0,0].set_ylabel(r"$R_\mathrm{infl}$/kpc")
    ax[1,0].set_ylabel(r"$R_\mathrm{min}$/kpc")
    ax[1,0].set_xlabel("Time/Myr")
    ax[1,1].set_xlabel("Time/Myr")
    subdirs = [f"{i:03d}" for i in range(num_subdirs)]
    funcargs = [(p, ax) for p in subdirs]
    if args.new:
        with mp.Pool(processes=num_subdirs) as pool:
            pool.starmap(_mphelper, funcargs)
    else:
        for p in subdirs:
            print(f"Reading {p}")
            data = bgs.utils.load_data(f"./pickle/early-{p}.pickle")
            plotter(ax, data, p, plot_median=False)
    ax[0,1].legend(title="Run", bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(os.path.join(bgs.FIGDIR, "merger-test/earlytimes/rinfl.png"))
    if not args.new:
        plt.show()
