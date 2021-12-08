import argparse
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Check the progress of an ongiong merger simulation.", allow_abbrev=False)
parser.add_argument(type=str, help="BH file", dest="file")
parser.add_argument("-b", "--bound", help="Plot bound points", dest="bound", action="store_true")
parser.add_argument("-o", "--orbit", help="Plot orbital parameters", dest="orbparams", action="store_true")
args = parser.parse_args()

#copy file so it can be read
filename, fileext = os.path.splitext(args.file)
new_filename = "{}_cp{}".format(filename, fileext)
shutil.copyfile(args.file, new_filename)

gyr = ketjugw.units.yr * 1e9
kpc = ketjugw.units.pc * 1e3

bh1, bh2, merged = cmf.analysis.get_bh_particles(new_filename)
if not merged:
    separation = cmf.mathematics.radial_separation(bh1.x/kpc, bh2.x/kpc)
    energy = ketjugw.orbital_energy(bh1, bh2)

    bound_points = np.diff(np.sign(energy), prepend=0) < 0
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(bh1.t/gyr, separation)
    ax[1].plot(bh1.t/gyr, energy)
    if args.bound and np.any(bound_points):
        ax[1].scatter(bh1.t[bound_points]/gyr, energy[bound_points], c="tab:red", zorder=10, marker=".")
        max_energy = np.max(energy)
        print("BHs become bound at:")
        for i, t in enumerate(bh1.t[bound_points]/gyr):
            ax[1].annotate(i, (t, 0), (t,max_energy), arrowprops={"arrowstyle":"->"}, horizontalalignment="center")
            print("{}: {:.3f} Gyr".format(i, t))
        
    ax[1].axhline(0, c="k", alpha=0.6)
    ax[0].set_yscale("log")
    ax[1].set_xlabel("t/Gyr")
    ax[0].set_ylabel("r/kpc")
    ax[1].set_ylabel("Energy")


if args.orbparams:
    bh1, bh2 = cmf.analysis.get_bound_binary(new_filename)
    op = ketjugw.orbital_parameters(bh1, bh2)
    fig2, ax2 = plt.subplots(2,1, sharex=True)
    ax2[0].set_ylabel("a/kpc")
    ax2[1].set_ylabel("e")
    ax2[1].set_xlabel("t/Gyr")
    ax2[0].semilogy(op["t"]/gyr, op["a_R"]/kpc)
    ax2[1].plot(op["t"]/gyr, op["e_t"])
plt.show()