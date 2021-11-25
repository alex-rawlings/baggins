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
parser.add_argument("-r", "--radius", type=float, help="radius to calculate density within [kpc]", dest="radius", default=0.1)
parser.add_argument("-t", "--time", type=float, help="plot times after this [Gyr]", dest="time", default=-1)
args = parser.parse_args()

if args.new:
    ball_radius = pygad.UnitScalar(args.radius, "kpc")
    snapfiles = cmf.utils.get_snapshots_in_dir(args.path)
    times = np.full_like(snapfiles, np.nan, dtype=float)
    diff_x = np.full((len(snapfiles),3), np.nan, dtype=float)
    diff_v = np.full_like(diff_x, np.nan, dtype=float)
    stellar_density = np.full_like(times, np.nan, dtype=float)

    for ind, snapfile in enumerate(snapfiles):
        print("Reading {:.2f}% done".format(ind/(len(snapfiles)-1)*100), end="\r")
        snap = pygad.Snapshot(snapfile)
        snap.to_physical_units()
        times[ind] = cmf.general.convert_gadget_time(snap)
        #ensure this is an isolated system
        assert len(snap.bh["ID"])==1, "The system must be isolated!"
        #get com motions
        xcom = cmf.analysis.get_com_of_each_galaxy(snap, verbose=False)
        diff_x[ind, :] = list(xcom.values())[0] - snap.bh["pos"]
        vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, verbose=False)
        diff_v[ind, :] = list(vcom.values())[0] - snap.bh["vel"]
        #get stellar density
        ball_mask = pygad.BallMask(ball_radius, center=list(xcom.values())[0])
        stellar_ball_mass = len(snap.stars[ball_mask])*snap.stars["mass"][0]
        stellar_density[ind] = cmf.mathematics.density_sphere(stellar_ball_mass, ball_radius)
        snap.delete_blocks()
    galaxy_name = args.path.rstrip("/").split("/")[-2]
    data_dict = {"times":times, "diff_x":diff_x, "diff_v":diff_v, "stellar_density": stellar_density, "ball_radius":args.radius, "galaxy_name":galaxy_name}
    savepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pickle/bh_perturb")
    os.makedirs(savepath, exist_ok=True)
    savefile = os.path.join(savepath, "{}_bhperturb.pickle".format(galaxy_name))
    cmf.utils.save_data(data_dict, savefile)
else:
    data_dict = cmf.utils.load_data(args.path)

#create a mask to limit to the desired time span
time_mask = data_dict["times"] > args.time
#determine Brownian magnitudes
displacement = cmf.mathematics.radial_separation(data_dict["diff_x"])
vel_mag = cmf.mathematics.radial_separation(data_dict["diff_v"])
#plot the displacement and velocity magnitudes, with a kernel-density estimate
p = sns.jointplot(x=displacement[time_mask], y=vel_mag[time_mask], kind="reg")
p.set_axis_labels("Radial Displacement [kpc]", "Velocity Magnitude [km/s]")
p.figure.suptitle(data_dict["galaxy_name"])
plt.subplots_adjust(left=0.1, bottom=0.07, top=0.95)
plt.savefig(os.path.join(cmf.FIGDIR, "brownian/{}_brownian.png".format(data_dict["galaxy_name"])))