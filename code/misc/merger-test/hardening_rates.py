import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pygad
import baggins as bgs
import ketjugw


class TimeEstimates:
    def __init__(self, orbit_params, start_idx, end_idx, Gps, H, K):
        self.a = {"median":None, "upper":None, "lower":None}
        self.e = {"median":None, "upper":None, "lower":None}
        self.t = {"median":None, "upper":None, "lower":None}
        self.op = orbit_params
        self.start_idx = start_idx
        self.end_idx = end_idx

        for k, q in zip(("median", "upper", "lower"), (0.5, 0.95, 0.05)):
            a0 = self.op["a_R"][self.start_idx]
            e0 = np.nanquantile(self.op["e_t"][self.start_idx:self.end_idx], q)
            self.t[k], self.a[k], self.e[k] = bgs.analysis.analytic_evolve_peters_quinlan(a0, e0, self.op["t"][start_idx], self.op["t"][-1], self.op["m0"][0], self.op["m1"][0], Gps, H, K)
    
    def plot(self, ax1, ax2, **kwargs):
        pc = ketjugw.units.pc
        myr = ketjugw.units.yr * 1e6
        for i, k in enumerate(self.a.keys()):
            if i==0:
                l = ax1.plot(self.t[k]/myr, self.a[k]/pc, ls="--", **kwargs)
            else:
                ax1.plot(self.t[k]/myr, self.a[k]/pc, ls=":", c=l[-1].get_color(), **kwargs)
            ax2.plot(self.t[k]/myr, self.e[k], ls=("--" if i%3==0 else ":"), c=l[-1].get_color(), **kwargs)


main_path = "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations_eta_0002/008/output"
bhfile = os.path.join(main_path, "ketju_bhs.hdf5")

snaplist = bgs.utils.get_snapshots_in_dir(main_path)

snap = pygad.Snapshot(snaplist[0], physical=True)
r_infl = list(bgs.analysis.influence_radius(snap).values())[0].in_units_of("pc") #in pc
r_hard = bgs.analysis.hardening_radius(snap.bh["mass"], r_infl)

bh1, bh2, merged = bgs.analysis.get_bound_binary(bhfile)
orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)
myr = ketjugw.units.yr * 1e6

r_infl_time = bgs.general.xval_of_quantity(r_infl, orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
r_hard_time = bgs.general.xval_of_quantity(r_hard, orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
r_hard_time_idx = np.argmax(r_hard_time < orbit_params["t"]/myr)
hard_snap_idx = bgs.analysis.snap_num_for_time(snaplist, r_hard_time, method="nearest")

snap = pygad.Snapshot(snaplist[hard_snap_idx], physical=True)

a_more_Xpc = np.argmax(orbit_params["a_R"]/ketjugw.units.pc<15)
time_a_more_Xpc = orbit_params["t"][a_more_Xpc]/myr
tspan = time_a_more_Xpc - r_hard_time
print("H determined over a span of {} Myr".format(tspan))

H, G_rho_per_sigma = bgs.analysis.linear_fit_get_H(orbit_params["t"]/ketjugw.units.yr, orbit_params["a_R"]/ketjugw.units.pc, r_hard_time*1e6, tspan*1e6, snap, r_infl, return_Gps=True)
mean_e = np.median(orbit_params["e_t"][r_hard_time_idx:a_more_Xpc])
print("e0: {:.3f}".format(mean_e))
print("G*rho/sigma: {:.3e}".format(G_rho_per_sigma))
print("Hardening rate: {:.4f}".format(H))
K = bgs.analysis.linear_fit_get_K(orbit_params["t"]/ketjugw.units.yr, orbit_params["e_t"], r_hard_time*1e6, tspan*1e6, H, G_rho_per_sigma, orbit_params["a_R"]/ketjugw.units.pc)
print("Eccentricity rate K: {:.4f}".format(K))

a_gr, a_gr_time = bgs.analysis.gravitational_radiation_radius(snap, r_infl, r_hard, r_hard_time, H, e=mean_e)
pq_estimates = TimeEstimates(orbit_params, r_hard_time_idx, a_more_Xpc, G_rho_per_sigma, H, K)
#peters_t, peters_a, peters_e = bgs.analysis.analytic_evolve_peters_quinlan(orbit_params, a_more_Xpc, H, K, G_rho_per_sigma)

print("Hardening radius a_h: {:.2e}".format(r_hard))
print("GR emission radius a_GR: {:.2e}".format(a_gr))


fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].set_ylabel("a/pc")
ax[1].set_xlabel("t/Myr")
ax[1].set_ylabel("e")
ax[0].semilogy(orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc)
ax[0].scatter(r_infl_time, r_infl, zorder=10, label=r"$r_\mathrm{inf}$")
sc = ax[0].scatter(r_hard_time, r_hard, zorder=10, label=r"$a_\mathrm{h}$")
ax[0].axvline(r_hard_time+tspan, c="tab:red")
ax[1].plot(orbit_params["t"]/myr, orbit_params["e_t"])
ax[0].scatter(a_gr_time, a_gr, zorder=10, label=r"$a_\mathrm{GR}$")
pq_estimates.plot(*ax)

"""ax[0].plot(peters_t/myr, peters_a/ketjugw.units.pc, ls="--", label="PQ", c=sc.get_facecolors())
ax[1].plot(peters_t/myr, peters_e, ls="--", c=sc.get_facecolors())"""

#create inset zoom axis
if False:
    axins = ax[0].inset_axes([0.5, 0.6, 0.4, 0.3])
    axins.semilogy(orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc)
    axins.set_xlim(252, 256)
    axins.set_ylim(30.5, 33.5)
    ax[0].indicate_inset_zoom(axins, edgecolor="black")
ax[0].legend()

fig2, ax2 = plt.subplots(1,1)
dadt, dedt = ketjugw.peters_derivatives(orbit_params["a_R"], orbit_params["e_t"], orbit_params["m0"], orbit_params["m1"])
ax2.semilogy(orbit_params["t"]/myr, -dadt)
plt.show()