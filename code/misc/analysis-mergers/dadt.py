import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d, median_filter, gaussian_filter1d
import ketjugw
from baggins.utils import get_ketjubhs_in_dir

ketjufiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/trail_blazer/H_2M_b-H_2M_c-30.000-2.000/output/ketju_bhs_cp.hdf5",
    ]
#ketjufiles = get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/trail_blazer/")

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
use_smoothing = True

for i, kf in enumerate(ketjufiles):
    if i > 5: break
    bhs = ketjugw.load_hdf5(kf)
    print("loaded")
    bhs = ketjugw.find_binaries(bhs, remove_unbound_gaps=True)
    print("binaries found")
    smooth_yrs = 1e4 # yr
    a_max = 6

    for bhid, bbh in bhs.items():
        op = ketjugw.orbital_parameters(*bbh)
        print("orbital parameters done")
        t = op["t"] / ketjugw.units.yr
        print(f"{(t[1]-t[0]):.2e}")
        smooth = smooth_yrs / (t[1]-t[0])
        print(f"Smoothing: {smooth:.2e}")
        a = gaussian_filter1d(op["a_R"], smooth, mode="nearest")
        e = gaussian_filter1d(op["e_t"], smooth, mode="nearest")

        # GW contribution
        dadt_gw, dedt_gw = ketjugw.peters_derivatives(op["a_R"], op["e_t"], bbh[0].m[0], bbh[1].m[0])
        a /= ketjugw.units.pc
        inv_a = 1 / a
        dadt_gw /= (ketjugw.units.pc / ketjugw.units.yr)
        inv_dadt_gw = -dadt_gw / (op["a_R"] / ketjugw.units.pc)**2
        if use_smoothing:
            dadt_gw = gaussian_filter1d(dadt_gw, smooth, mode="nearest")
            inv_dadt_gw = gaussian_filter1d(inv_dadt_gw, smooth, mode="nearest")

        mask = a < a_max
        N = np.sum(mask)
        print(f"Valid: {N}")
        if N  == 0:
            print(f"Min a: {np.min(a):.2e}")
            continue

        # total
        dadt = np.gradient(op["a_R"]/ketjugw.units.pc, t)
        inv_dadt = np.gradient(ketjugw.units.pc/op["a_R"], t)
        if use_smoothing:
            dadt = gaussian_filter1d(dadt, smooth, mode="nearest")
            inv_dadt = gaussian_filter1d(inv_dadt, smooth, mode="nearest")
        dadt_h = dadt - dadt_gw
        inv_dadt_h = 1 / dadt_h
        #dadt_h = gaussian_filter1d(dadt_h, smooth, mode="nearest")

        # integrate quantities
        a_gw = cumulative_trapezoid(dadt_gw, t, initial=0)
        '''a_gw_samples = np.full(100, np.nan)
        tsamples = np.arange(-10000, -10, 100)
        for i, idx in enumerate(tsamples):
            a_gw_samples[i] = ketjugw.peters_evolution(a0=a[idx], e0=e[idx], m1=bbh[0].m[0], m2=bbh[1].m[0], ts=t[mask][idx:])[0]
        a_gw = np.interp(t[mask], tsamples, a_gw_samples)'''
        inv_a_gw = cumulative_trapezoid(inv_dadt_gw, t, initial=0)
        a_h = a - a_gw
        inv_a_h = 1 / a_h

        # plot
        ax[0].axvline(t[mask][0], c="k", ls=":")
        ax[0].plot(t, op["a_R"]/ketjugw.units.pc, alpha=0.8)
        ax[0].plot(t, a, alpha=0.8)
        ax[0].plot(t, a_h, alpha=0.8)
        ax[0].plot(t, a_gw, alpha=0.8)
        ax[0].set_yscale("symlog", linthresh=1e-3)

        ax[1].plot(t[mask], dadt[mask], alpha=0.8)
        ax[1].plot(t[mask], dadt_gw[mask], alpha=0.8)
        #ax[1].plot(t[mask], dadt_h[mask], alpha=0.8)
        ax[1].set_yscale("symlog", linthresh=1e-8)

        ax[2].plot(
            #-a_h[mask], a_gw[mask],
            #inv_a_h[mask], inv_a_gw[mask],
            #dadt_h[mask], dadt_gw[mask],
            #inv_dadt_h[mask], inv_dadt_gw[mask],
            #t[mask], -a_gw[mask] / a[mask],
            dadt[mask], dadt_gw[mask],
            marker="o", markevery=[-1]
        )
        ax[2].set_xscale("symlog", linthresh=1e-8)
        ax[2].set_yscale("symlog", linthresh=1e-8)

xlim = ax[2].get_xlim()
ylim = ax[2].get_ylim()
ax[2].plot(xlim, xlim, c="k", ls=":", zorder=0.2)
ax[2].set_xlim(*xlim)
ax[2].set_ylim(*ylim)

#ax[2].set_xlabel("-da/dt|H [pc/yr]")
#ax[2].set_ylabel("-da/dt|GW [pc/yr]")
plt.savefig("dadt.png")
