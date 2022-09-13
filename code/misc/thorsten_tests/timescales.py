import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf
from local_funcs import HMQ_generator, get_hard_timespan


data_dirs = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.001",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.005",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.05",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.1"
]

fig, ax = plt.subplots(1,2, sharex="all")
ax[0].set_xlabel("t/Myr")
ax[1].set_ylabel(r"$R_\mathrm{infl}$/kpc")
ax[0].set_xlabel("t/Myr")
ax[1].set_ylabel(r"$a_\mathrm{hard}$/kpc")

fig2, ax2 = plt.subplots(1,1)
ax2.set_xlabel("Simulation")
ax2.set_ylabel("Time Binary Hard For (Myr)")


for (hmq, alpha, count, f) in HMQ_generator(data_dirs, True):
    t = hmq.time_of_snapshot
    rinfl = hmq.influence_radius
    ahard = hmq.hardening_radius
    if hmq.merger_remnant["merged"]:
        last_idx = -1 if count != 24 else -12
        t = t[:last_idx]
        rinfl = rinfl[:last_idx]
        ahard = ahard[:last_idx]

    if count < 10:
        ls = "-"
    elif count < 20:
        ls = "--"
    elif count < 30:
        ls = ":"
    else:
        ls = "-."
    ax[0].plot(t, rinfl, ls=ls, alpha=alpha, label=(count if count<20 else ""))
    ax[1].plot(t, ahard, ls=ls, alpha=alpha, label=(count if count>19 else ""))

    ax2.scatter(count, get_hard_timespan(hmq.binary_time, hmq.semimajor_axis, t, ahard), alpha=alpha)
ax[0].legend(fontsize="x-small")
ax[1].legend(fontsize="x-small")

cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"other_tests/thorsten/AC_radii.png"), fig=fig)
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"other_tests/thorsten/AC_hardtime.png"), fig=fig2)
plt.show() 