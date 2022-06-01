import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf
import pygad
from ketjugw.units import pc, yr


def get_and_centre_bhs(k, bound=False):
    if bound:
        bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
    else:
        bh1, bh2, merged = cmf.analysis.get_bh_particles(k)
    mass_sum = np.atleast_2d(bh1.m + bh2.m).T
    xcom = (np.atleast_2d(bh1.m).T * bh1.x + np.atleast_2d(bh2.m).T * bh2.x) / mass_sum
    vcom = (np.atleast_2d(bh1.m).T * bh1.v + np.atleast_2d(bh2.m).T * bh2.v) / mass_sum
    vcom
    bh1.x -= xcom
    bh1.v -= vcom
    bh2.x -= xcom
    bh2.v -= vcom

    return bh1, bh2


mainpath = "/scratch/pjohanss/arawling/collisionless_merger/high-time-output/A-C-3.0-0.05-H"

ketjufiles = cmf.utils.get_ketjubhs_in_dir(mainpath)


fig, ax = plt.subplots(1,2, sharex="all")
fig2, ax2 = plt.subplots(3,1, sharex="all", sharey="all")
ls = ["-", ":"]
alpha = [0.3, 0.8]

for i, k in enumerate(ketjufiles):
    print(f"{i:03d}")
    bh1, bh2 = get_and_centre_bhs(k)
    bhb1, bhb2 = get_and_centre_bhs(k, bound=True)
    for j, (x,y) in enumerate(zip((0,0), (1,2))):
        l = ax[j].plot(bh1.x[:,x]/pc, bh1.x[:,y]/pc, ls=ls[0], alpha=alpha[0])
        ax[j].plot(bh2.x[:,x]/pc, bh2.x[:,y]/pc, ls=ls[1], c=l[-1].get_color(), alpha=alpha[0])
        ax[j].plot(bhb1.x[:,x]/pc, bhb1.x[:,y]/pc, ls=ls[1], c=l[-1].get_color(), alpha=alpha[1])
        ax[j].plot(bhb2.x[:,x]/pc, bhb2.x[:,y]/pc, ls=ls[1], c=l[-1].get_color(), alpha=alpha[1])
    L1 = np.cross(bh1.x, bh1.v)
    L2 = np.cross(bh2.x, bh2.v)
    L1b = np.cross(bhb1.x, bhb1.v)
    L2b = np.cross(bhb2.x, bhb2.v)
    for j in range(3):
        l = ax2[j].plot(bh1.t/yr, L1[:,j], ls=ls[0], alpha=alpha[0])
        ax2[j].plot(bh1.t/yr, L2[:,j], ls=ls[1], c=l[-1].get_color(), alpha=alpha[0])
        ax2[j].plot(bhb1.t/yr, L1b[:,j], ls=ls[1], c=l[-1].get_color(), alpha=alpha[1], marker="s", markevery=[0])
        ax2[j].plot(bhb1.t/yr, L2b[:,j], ls=ls[1], c=l[-1].get_color(), alpha=alpha[1], marker="s", markevery=[0])
plt.show()