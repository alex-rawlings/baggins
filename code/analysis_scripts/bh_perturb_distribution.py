import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pygad
import cm_functions as cmf


#set up command line arguments
parser = argparse.ArgumentParser(description="Determine the deviation of the BH from the CoM for an isolated galaxy", allow_abbrev=False)
parser.add_argument(type=str, help="path to snapshot directory or previous dataset", dest="path")
parser.add_argument("-N", "--new", help="analyse a new dataset", dest="new", action="store_true")
args = parser.parse_args()

if args.new:
    snapfiles = cmf.utils.get_snapshots_in_dir(args.path)
    diff_x = np.full((len(snapfiles),3), np.nan, dtype=float)
    diff_v = np.full((len(snapfiles),3), np.nan, dtype=float)

    for ind, snapfile in enumerate(snapfiles):
        print("Reading {:.2f}% done".format(ind/(len(snapfiles)-1)*100), end="\r")
        snap = pygad.Snapshot(snapfile)
        snap.to_physical_units()
        #ensure this is an isolated system
        assert len(snap.bh["ID"])==1, "The system must be isolated!"
        xcom = cmf.analysis.get_com_of_each_galaxy(snap, verbose=False)
        diff_x[ind, :] = list(xcom.values())[0] - snap.bh["pos"]
        vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, verbose=False)
        diff_v[ind, :] = list(vcom.values())[0] - snap.bh["vel"]
        snap.delete_blocks()
    data_dict = {"diff_x":diff_x, "diff_v":diff_v}
    savepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pickle/bh_perturb")
    os.makedirs(savepath, exist_ok=True)
    savefile = os.path.join(savepath, "{}_bhperturb.pickle".format(args.path.rstrip("/").split("/")[-2]))
    cmf.utils.save_data(data_dict, savefile)
else:
    data_dict = cmf.utils.load_data("perturb.pickle")

data_array = np.hstack((data_dict["diff_x"], data_dict["diff_v"]))
#bootstrap
samples, means = cmf.mathematics.smooth_bootstrap(data_array)
fig, ax = plt.subplots(3,2, sharex="col", sharey="col", figsize=(4, 6))
ax[2,0].set_xlabel("kpc")
ax[2,1].set_xlabel("km/s")
labvals = ["x", "y", "z", "vx", "vy", "vz"]
ax = np.concatenate(ax.T)
for i in range(6):
    ax[i].hist(samples[:,i], density=True)
    ax[i].axvline(means[i], c="tab:red", label=("BS" if i==0 else ""))
    ax[i].axvline(np.std(data_array[:,i]), c="tab:orange", label=("Obs." if i==0 else ""))
    ax[i].text(0.1, 0.9, labvals[i], transform=ax[i].transAxes)
ax[0].legend()
plt.show()
"""
diff_x, diff_v = data_dict.values()
fig, ax = plt.subplots(3,2, sharex="col", sharey="col", figsize=(4, 7))
for i in range(3):
    ax[i, 0].hist(diff_x[:, i], density=True, alpha=0.6)
    ax[i, 0].text(0.1, 0.9, r"$\sigma$: {:.3f}".format(np.nanstd(diff_x[:, i])), transform=ax[i,0].transAxes)
    ax[i, 1].hist(diff_v[:, i], density=True, alpha=0.6)
    ax[i, 1].text(0.1, 0.9, r"$\sigma$: {:.3f}".format(np.nanstd(diff_v[:, i])), transform=ax[i,1].transAxes)
plt.show()
"""