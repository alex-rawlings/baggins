import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import os
import pygad
import cm_functions as cmf


#set up command line arguments
parser = argparse.ArgumentParser(description="Determine the deviation of the BH from the CoM for an isolated galaxy", allow_abbrev=False)
parser.add_argument(type=str, help="path to snapshot directory or previous dataset", dest="path")
parser.add_argument("-n", "--new", help="analyse a new dataset", dest="new", action="store_true")
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
    galaxy_name = args.path.rstrip("/").split("/")[-2]
    data_dict = {"diff_x":diff_x, "diff_v":diff_v, "galaxy_name":galaxy_name}
    savepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pickle/bh_perturb")
    os.makedirs(savepath, exist_ok=True)
    savefile = os.path.join(savepath, "{}_bhperturb.pickle".format(galaxy_name))
    cmf.utils.save_data(data_dict, savefile)
else:
    data_dict = cmf.utils.load_data(args.path)

displacement = cmf.mathematics.radial_separation(data_dict["diff_x"])
vel_mag = cmf.mathematics.radial_separation(data_dict["diff_v"])
#plot the displacement and velocity magnitudes, with a kernel-density estimate
p = sns.jointplot(x=displacement, y=vel_mag, kind="reg")
p.set_axis_labels("Radial Displacement [kpc]", "Velocity Magnitude [km/s]")
p.figure.suptitle(data_dict["galaxy_name"])
plt.subplots_adjust(left=0.1, bottom=0.07, top=0.95)
plt.savefig(os.path.join(cmf.FIGDIR, "brownian/{}_brownian.png".format(data_dict["galaxy_name"])))