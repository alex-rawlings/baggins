import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
from local_funcs import HMQ_generator


parser = argparse.ArgumentParser(description="Thorsten plots", allow_abbrev=False)
parser.add_argument("-m", "--merged", help="highlight mergers", 
action="store_true", dest="merged")
parser.add_argument("-k", "--kicked", help="highlight kicks > 1000 km/s", type=str, choices=["None", "Low", "High"], default="None", dest="kicked")
args = parser.parse_args()

data_dirs = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.001",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.005",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.05",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.1"
]

fig, ax = plt.subplots(1,2, sharex="all", sharey="all")
fig2, ax2 = plt.subplots(1,2, sharex="all", sharey="all")
fig3, ax3 = plt.subplots(1,2, sharex="all", sharey="all")
axins = []
for axi in ax:
    axi.set_xlabel(r"$r$/kpc")
    _axins = axi.inset_axes([0.2,0.1,0.5,0.5])
    axins.append(_axins)
ax[0].set_ylabel(r"$\Sigma(r)/($M$_\odot$kpc$^{-2})$")
for axi in ax2:
    axi.set_xlabel(r"$r$/kpc")
ax2[0].set_ylabel(r"$\beta(r)$")

hmq_gen = HMQ_generator(data_dirs, args.merged, args.kicked)
for (hmq, alpha, count, f) in hmq_gen:
    r = bgs.mathematics.get_histogram_bin_centres(hmq.radial_edges)
    Sigma = np.nanmedian(list(hmq.projected_mass_density.values())[-1], axis=0)
    beta = list(hmq.velocity_anisotropy.values())[-1]

    ax_i = count//20
    ls = "-" if (count//10)%2 ==0 else ":"
    ax[ax_i].loglog(r, Sigma, ls=ls, label=count, alpha=alpha)
    axins[ax_i].loglog(r, Sigma, ls=ls, alpha=alpha)
    try:
        ax2[ax_i].semilogx(r, beta, ls=ls, alpha=alpha, label=count)
        ax3[ax_i].semilogy(np.cumsum(-beta), np.cumsum(Sigma), ls=ls, alpha=alpha, label=count)
    except ValueError:
        print(f"Beta length for {f} ({count}): {len(beta)}")
        # plot with zero alpha, so colour scheme matches between plots
        ax2[ax_i].semilogx(beta, ls=ls, alpha=0, label=count)
        ax3[ax_i].semilogy(0,0, ls=ls, alpha=0, label=count)

for axi in ax:
    axi.legend(fontsize="x-small", loc="upper right")
for axi in ax2:
    axi.legend(fontsize="x-small", loc="lower right")
for (_axins, axi) in zip(axins, ax):
    _axins.set_xlim(0.09, 2)
    _axins.set_ylim(2e9, 4.5e9)
    axi.indicate_inset_zoom(_axins)

mflag = "mergers_shown" if args.merged else "mergers_hidden"
if args.kicked == "None":
    kflag = "all"
elif args.kicked == "High":
    kflag = "high_kick"
else:
    kflag = "low_kick"

bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"other_tests/thorsten/AC_density_{mflag}_{kflag}.png"), fig=fig)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"other_tests/thorsten/AC_beta_{mflag}_{kflag}.png"), fig=fig2)
plt.close(fig)
plt.close(fig2)
plt.show()