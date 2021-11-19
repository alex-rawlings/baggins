import argparse
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Check the progress of an ongiong merger simulation.", allow_abbrev=False)
parser.add_argument(type=str, help="BH file", dest="file")
args = parser.parse_args()

#copy file so it can be read
filename, fileext = os.path.splitext(args.file)
new_filename = "{}_cp{}".format(filename, fileext)
shutil.copyfile(args.file, new_filename)

myr = ketjugw.units.yr * 1e6
kpc = ketjugw.units.pc * 1e3

bhs = ketjugw.data_input.load_hdf5(new_filename)
bh1, bh2 = bhs.values()
separation = cmf.mathematics.radial_separation(bh1.x/kpc, bh2.x/kpc)
energy = ketjugw.orbital_energy(bh1, bh2)

bound_points = np.diff(np.sign(energy), prepend=0) < 0
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(bh1.t/myr, separation)
ax[1].plot(bh1.t/myr, energy)
if np.any(bound_points):
    ax[1].scatter(bh1.t[bound_points]/myr, energy[bound_points], c="tab:red", zorder=10, marker=".")
    max_energy = np.max(energy)
    print("BHs become bound at:")
    for i, t in enumerate(bh1.t[bound_points]/myr):
        ax[1].annotate(i, (t, 0), (t,max_energy), arrowprops={"arrowstyle":"->"}, horizontalalignment="center")
        print("{}: {:.2f} Myr".format(i, t))
    
ax[1].axhline(0, c="k", alpha=0.6)
ax[0].set_yscale("log")
ax[1].set_xlabel("t/Myr")
ax[0].set_ylabel("r/kpc")
ax[1].set_ylabel("Energy")
plt.show()