import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py
import baggins as bgs
import ketjugw


def move_bh_to_com(bh1, bh2):
    m1 = bh1.m[:, np.newaxis]
    m2 = bh2.m[:, np.newaxis]
    sum_m = m1+m2
    xcom = (m1 * bh1.x + m2 * bh2.x)/sum_m
    vcom = (m1 * bh1.v + m2 * bh2.v)/sum_m
    def _move(bh):
        bhnew = copy.copy(bh)
        bhnew.x -= xcom
        bhnew.v -= vcom
        return bhnew
    bh1new = _move(bh1)
    bh2new = _move(bh2)
    return bh1new, bh2new


parser = argparse.ArgumentParser(description="Quickly check SMBH binary parameters", allow_abbrev=False)
parser.add_argument(type=str, help="path to directory", dest="path")
#parser.add_argument("-m", "--masking", type=float, help="mask to times less than this (Myr)", default=None, dest="mask")
args = parser.parse_args()

myr = 1e6 * ketjugw.units.yr

ketjufiles = bgs.utils.get_ketjubhs_in_dir(args.path)
fig, ax = plt.subplots(1,1)
ax.set_xlabel("t/Myr")
ax.set_ylabel("r/pc")
ax.set_yscale("log")

alpha = (0.7, 0.1)

for i, k in enumerate(ketjufiles):
    print(k)
    bh1, bh2, merged = bgs.analysis.get_bh_particles(k)
    bh1b, bh2b, merged = bgs.analysis.get_bound_binary(k)
    unbound_mask = bh1.t < bh1b.t[0]
    for j, (b1, b2, a) in enumerate(zip((bh1[unbound_mask], bh1b), (bh2[unbound_mask], bh2b), alpha)):
        b1, b2 = move_bh_to_com(b1, b2)
        bh1_r = bgs.mathematics.radial_separation(b1.x)/ketjugw.units.pc
        bh2_r = bgs.mathematics.radial_separation(b2.x)/ketjugw.units.pc
        if j ==0:
            l = ax.plot(b1.t/myr, bh1_r, label=f"{k.split('/')[-3]}", alpha=a)
        else:
            ax.plot(b1.t/myr, bh1_r, c=l[-1].get_color(), alpha=a)
        ax.plot(b2.t/myr, bh2_r, c=l[-1].get_color(), ls=":", alpha=a)
ax.legend()
plt.show()