import numpy as np
import scipy.optimize
import os.path
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf


low_e = "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/stars_only_low_e"
high_e = "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/stars_only_high_e"
N_stars = [128e3, 256e3, 512e3, 2048e3]
func = lambda x, k: k/np.sqrt(x+k**2)

myr = ketjugw.units.yr * 1e6
avg_period = 10 * 0.304 * myr

# plotting sequence for expected relation
xseq = np.geomspace(0.9*np.min(N_stars), 1.1*np.max(N_stars), 500)

fig, ax = plt.subplots(1,1)
ax.set_xlabel("Num Stars")
ax.set_ylabel("Eccentricity variance")
ax.set_xscale("log")
ax.set_yscale("log")

for d, lab in zip((low_e, high_e), (r"$e_0=0.9$", "$e_0>0.99$")):
    ecc_std = []
    for res in ("PR", "LR", "MR", "HR"):
        ketju_files = cmf.utils.get_ketjubhs_in_dir(os.path.join(d, res))
        ecc_std_temp = []
        for k in ketju_files:
            try:
                bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
            except:
                print(f"No bound binary in {k}, skipping")
                continue
            op = ketjugw.orbital_parameters(bh1, bh2)
            mask = op["t"]/myr < op["t"][0]/myr + avg_period
            ecc_std_temp.append(np.nanmean(op["e_t"][mask]))
        ecc_std.append(np.nanstd(ecc_std_temp))
    # fit the functional form
    popt, pcov = scipy.optimize.curve_fit(func, N_stars, ecc_std)
    print(f"Optimal parameters: {popt}")
    ax.scatter(N_stars, ecc_std, label=lab, zorder=10, ec="k", lw=0.5, s=100)
    # overlay scaling
    ax.plot(xseq, func(xseq, popt), lw=2, label=f"$k({lab})={popt[0]:.2f}$")
ax.legend()
ax.set_title("Star-only Hernquist mergers")
ax.text(0.1, 0.1, r"$f(x;k)=\frac{k}{\sqrt{x+k^2}}$", transform=ax.transAxes, fontsize=20)
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, "nasim_style_plot.png"))
plt.show()