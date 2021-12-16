import os.path
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf
import numpy as np
import scipy.optimize


myr = ketjugw.units.yr * 1e6
err_level = 0.05


main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers"
subdirs = [
    "D-E-3.0-0.001/perturbations",
    "D-E-3.0-0.005/perturbations",
    "D-E-3.0-0.1/perturbations",
    "D-E-3.0-1.0/perturbations"
]
cols = cmf.plotting.mplColours()
alpha = 0.6
bins = np.linspace(0, 1, 50)
bins2 = np.linspace(0, 1, 20)

fig, ax = plt.subplots(2,1, sharex="all")
fig2, ax2 = plt.subplots(1,1)

for i, subdir in enumerate(subdirs):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(os.path.join(main_path, subdir), file_name="ketju_bhs_cp.hdf5")
    peak_e = np.full_like(ketju_files, np.nan, dtype=float)
    label = r"$\eta$={}.{}".format(subdir[0], subdir[1:])
    for j, ketjufile in enumerate(ketju_files):
        print("Reading: {}".format(ketjufile))
        bh1, bh2, merged = cmf.analysis.get_bound_binary(ketjufile)
        op = ketjugw.orbital_parameters(bh1, bh2)
        ax[0].semilogy(op["t"]/myr, op["a_R"]/ketjugw.units.pc, c=cols[i], label=(label if j==0 else ""), alpha=alpha)
        ax[1].plot(op["t"]/myr, op["e_t"], c=cols[i], alpha=alpha)
        hval, hbins = np.histogram(op["e_t"], bins=bins)
        bincentres = cmf.mathematics.get_histogram_bin_centres(hbins)
        peak_e[j] = bincentres[np.argmax(hval)]
    pe_vals, pe_bins = np.histogram(peak_e, bins=bins2)
    ax2.scatter(cmf.mathematics.get_histogram_bin_centres(pe_bins), pe_vals+i*0.1, zorder=10, label=subdir.split("/")[0])
ax[0].legend()
ax2.legend()
ax2.set_ylabel("Number of Runs")
for i in range(1, 6):
    ax2.axhline(i, c="k", alpha=0.4)
ax[0].set_ylabel("a/pc")
ax[1].set_xlabel("t/Myr")
ax[1].set_ylabel("e")
ax2.set_xlabel("peak e")
plt.show()


quit()

main_path = "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations/"
data_file = "output/ketju_bhs_cp.hdf5"

#fig, ax = plt.subplots(3,1, sharex="all")
fig, ax = plt.subplots(2,1, sharex="all")
ax[0].set_ylabel("pc/a")
ax[1].set_ylabel("e")
ax[1].set_xlabel("t/Myr")

#ax[0].set_xscale("log")
for i in (8,9,):
    label = "{:03d}".format(i)
    bhfile = os.path.join(main_path, label, data_file)
    bh1, bh2, merged = cmf.analysis.get_bound_binary(bhfile)
    op = ketjugw.orbital_parameters(bh1, bh2)
    gwidx, gwtime = cmf.analysis.find_where_gw_dominate(op, err_level)
    if merged: gwdix=-1
    """for j in range(3):
        ax[j].plot(bh1.t/myr, op["plane_normal"][:,j], label=label, alpha=(0.9 if merged else 0.3))"""
    """gwidx, gwtime = cmf.analysis.find_where_gw_dominate(op, err_level)
    if merged: gwdix = -1
    ax[0].hist(np.log10(op["a_R"][:gwidx]/ketjugw.units.pc), np.linspace(-2.5, 3, 500), alpha=(0.9 if merged else 0.3), label=label, histtype="step")
    ax[1].hist(op["e_t"][:gwidx], np.linspace(0, 1, 100), alpha=(0.9 if merged else 0.3), histtype="step")"""
    ts = op["t"][:gwidx]/myr
    inv_a = 1/(op["a_R"][:gwidx]/ketjugw.units.pc)
    f = lambda x, a, b: a*x+b
    popt, pcov = scipy.optimize.curve_fit(f, ts, inv_a)
    ax[0].plot(ts, inv_a, label=label, alpha=(0.9 if merged else 0.3))
    ax[0].plot(ts, f(ts, *popt))
    #ax[1].semilogy(ts, np.abs(np.gradient(inv_a)))
    #x[1].plot(ts, op["e_t"][:gwidx], alpha=(0.9 if merged else 0.3))
ax[0].legend(fontsize="x-small")
plt.show()