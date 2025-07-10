import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs

# core radius
core_rad = 0.58

def threshold_dist(x, core_sig=270, upper_vk=1080):
    y = np.atleast_1d(3.21e-2 * x - 11.5)
    y[y < core_rad] = core_rad
    y[x < core_sig] = 1000
    y[x > upper_vk] = 1000
    return y


fig, ax = plt.subplots(1, 2, sharex="all")
fig.set_figwidth(2*fig.get_figwidth())
bins = np.geomspace(5e-2, 30, 31)

for axi in ax:
    axi.axvline(core_rad, c="k", ls=":", label="core radius")
    axi.grid(alpha=0.3)
    axi.set_xlabel(r"r$_\bullet$/kpc")
ax[0].set_ylabel("PDF")
ax[1].set_ylabel(r"P(r$_\bullet$>r)")

for i, kv in enumerate((900, 660, 540, 420)):
    print(f"Doing {kv}")
    kf = bgs.utils.get_ketjubhs_in_dir(f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kv:04d}")[0]

    bh = bgs.analysis.get_bh_after_merger(kf)
    bh.x /= bgs.general.units.kpc
    bh.t /= bgs.general.units.Gyr
    bh = bh[bh.t < 3]

    r = bgs.mathematics.radial_separation(bh.x)

    ecdf = bgs.mathematics.EmpiricalCDF(r)

    h = ax[0].hist(r, bins=bins, density=True, label=kv, alpha=1, histtype="step", lw=2)
    #c = h[-1].patches[0].get_facecolor()
    c = h[-1][-1].get_edgecolor()
    for axi in ax:
        axi.axvline(threshold_dist(kv), label="Detectable threshold" if i==0 else "", c=c)

    ecdf.plot(ax=ax[1], npoints=1000, survival=True, lw=2, c=c)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].legend()
ax[1].set_ylim(0, 1)

bgs.plotting.savefig("radial_dist.png")