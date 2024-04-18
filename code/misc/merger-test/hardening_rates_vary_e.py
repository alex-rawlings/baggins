import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pygad
import baggins as bgs
import ketjugw


def analytic_evolve_peters_quinlan(orbit_params, idx0, H, K, Gps, idx1=None, e0=0.9):
    if idx1 is None: idx1 = idx0+1
    # TODO errenous results for extremely high values of e -> how to handle??
    a0 = np.mean(orbit_params["a_R"][idx0:idx1])
    #e0 = 0.84 #np.mean(orbit_params["e_t"][idx0:idx1])
    t0, tf = orbit_params["t"][[idx0, -1]]
    m1 = orbit_params["m0"][idx0]
    m2 = orbit_params["m1"][idx0]

    #convert Gps to units used by ketjugw
    Gps = Gps / (ketjugw.units.pc * ketjugw.units.yr)

    def quinlan_derivatives(a, e, m1, m2):
        dadt = -a**2 * H * Gps
        dedt = -dadt / a * K
        return dadt, dedt

    propagate_time = 8*(tf-t0) + t0
    af, ef, _,_, tf = ketjugw.orbit.peters_evolution(a0, e0, m1, m2, (t0, propagate_time, 5), ext_derivs=quinlan_derivatives)
    return tf, af, ef

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
mean_e = np.mean(orbit_params["e_t"][r_hard_time_idx:a_more_Xpc])
print("e0: {:.3f}".format(mean_e))
print("G*rho/sigma: {:.3e}".format(G_rho_per_sigma))
print("Hardening rate: {:.4f}".format(H))
K = bgs.analysis.linear_fit_get_K(orbit_params["t"]/ketjugw.units.yr, orbit_params["e_t"], r_hard_time*1e6, tspan*1e6, H, G_rho_per_sigma, orbit_params["a_R"]/ketjugw.units.pc)
print("Eccentricity rate K: {:.4f}".format(K))


fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].set_ylabel("a/pc")
ax[1].set_xlabel("t/Myr")
ax[1].set_ylabel("e")
ax[0].semilogy(orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc)
ax[0].scatter(r_infl_time, r_infl, zorder=10, label=r"$r_\mathrm{inf}$")
sc = ax[0].scatter(r_hard_time, r_hard, zorder=10, label=r"$a_\mathrm{h}$")
ax[0].axvline(r_hard_time+tspan, c="tab:red")
ax[1].plot(orbit_params["t"]/myr, orbit_params["e_t"])
for e0 in (0.8, 0.85, 0.9, 0.95):
    a_gr, a_gr_time = bgs.analysis.gravitational_radiation_radius(snap, r_infl, H, e=e0)
    sc1 = ax[0].scatter(a_gr_time, a_gr, zorder=10)
    peters_t, peters_a, peters_e = bgs.analysis.analytic_evolve_peters_quinlan(orbit_params, r_hard_time_idx, H, K, G_rho_per_sigma, idx1=None, e0=e0)
    ax[0].plot(peters_t/myr, peters_a/ketjugw.units.pc, ls="--", label=r"$e_0=${}".format(e0), c=sc1.get_facecolors())
    ax[1].plot(peters_t/myr, peters_e, ls="--", label=r"$e_0=${}".format(e0), c=sc1.get_facecolors())

#create inset zoom axis
if False:
    axins = ax[0].inset_axes([0.5, 0.6, 0.4, 0.3])
    axins.semilogy(orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc)
    axins.set_xlim(252, 256)
    axins.set_ylim(30.5, 33.5)
    ax[0].indicate_inset_zoom(axins, edgecolor="black")
ax[0].legend()
plt.show()