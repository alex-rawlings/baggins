import argparse
import numpy as np
import matplotlib.pyplot as plt
import ketjugw

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(type=str, help="bh file", dest="file")
args = parser.parse_args()

bh1, bh2 = ketjugw.data_input.load_hdf5(args.file).values()

myr = ketjugw.units.yr * 1e6

plt.xlabel("Array Index")
plt.ylabel("BH Particle Time (Myr)")
plt.plot(bh1.t/myr, "-o", markevery=10000)
plt.plot(bh2.t/myr, "-o", markevery=10000)
plt.tight_layout()
plt.show()