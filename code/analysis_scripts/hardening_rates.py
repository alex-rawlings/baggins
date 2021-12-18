import argparse
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf
import ketjugw


parser = argparse.ArgumentParser(description="Compare the analytical hardening rate to ketju output", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="path")
parser.add_argument(type=str, help="perturbation number", dest="num")
parser.add_argument("-r", "--radiusgw", type=float, help="Radius [pc] above which GW emission expected to be negligible", dest="rgw", default=15)
args = parser.parse_args()



class TimeEstimates:
    def __init__(self, orbit_params, start_idx, end_idx, Gps, H, K):
        self.a = {"median":None, "upper":None, "lower":None}
        self.e = {"median":None, "upper":None, "lower":None}
        self.t = {"median":None, "upper":None, "lower":None}
        self.op = orbit_params
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.quantiles = (0.5, 0.95, 0.05)

        for k, q in zip(("median", "upper", "lower"), self.quantiles):
            a0 = self.op["a_R"][self.start_idx]
            e0 = np.nanquantile(self.op["e_t"][self.start_idx:self.end_idx], q)
            self.t[k], self.a[k], self.e[k] = cmf.analysis.analytic_evolve_peters_quinlan(a0, e0, self.op["t"][start_idx], self.op["t"][-1], self.op["m0"][0], self.op["m1"][0], Gps, H, K)
    
    def plot(self, ax1, ax2, toffset=0, **kwargs):
        pc = ketjugw.units.pc
        myr = ketjugw.units.yr * 1e6
        for i, (k,q) in enumerate(zip(self.a.keys(), self.quantiles)):
            if i==0:
                l = ax1.plot(self.t[k]/myr+toffset, self.a[k]/pc, ls="--", label="{:.2f} quantile".format(q), **kwargs)
            else:
                ax1.plot(self.t[k]/myr+toffset, self.a[k]/pc, ls=":", c=l[-1].get_color(), label="{:.2f} quantile".format(q), **kwargs)
            ax2.plot(self.t[k]/myr+toffset, self.e[k], ls=("--" if i%3==0 else ":"), c=l[-1].get_color(), **kwargs)

pfv = cmf.utils.read_parameters(args.path)
data_path = os.path.join(pfv.full_save_location, pfv.perturbSubDir, args.num, "output")

bhfile = os.path.join(data_path, "ketju_bhs.hdf5")
snaplist = cmf.utils.get_snapshots_in_dir(data_path)

#copy file so it can be read
filename, fileext = os.path.splitext(bhfile)
new_bhfile = "{}_cp{}".format(filename, fileext)
shutil.copyfile(bhfile, new_bhfile)

#determine the influence and hardening radii
snap = pygad.Snapshot(snaplist[0], physical=True)
time_offset = pfv.perturbTime * 1000 #in Myr
print("Perturbation applied at {:.1f} Myr".format(time_offset))
r_infl = list(cmf.analysis.influence_radius(snap).values())[0].in_units_of("pc") #in pc
r_hard = cmf.analysis.hardening_radius(snap.bh["mass"], r_infl)

#get BH objects
bh1, bh2, merged = cmf.analysis.get_bound_binary(new_bhfile)
orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)
myr = ketjugw.units.yr * 1e6

#determine when the above radii occur
r_infl_time = cmf.general.xval_of_quantity(r_infl, orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
r_hard_time = cmf.general.xval_of_quantity(r_hard, orbit_params["t"]/myr, orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
r_hard_time_idx = np.argmax(r_hard_time < orbit_params["t"]/myr)
hard_snap_idx = cmf.analysis.snap_num_for_time(snaplist, r_hard_time, method="nearest")

#get the snap from which inner density, sigma is found
snap = pygad.Snapshot(snaplist[hard_snap_idx], physical=True)


a_more_Xpc = np.argmax(orbit_params["a_R"]/ketjugw.units.pc<args.rgw)
time_a_more_Xpc = orbit_params["t"][a_more_Xpc]/myr
tspan = time_a_more_Xpc - r_hard_time
print("H determined over a span of {:.3f} Myr".format(tspan))

#determine hardening constants -- times are in years
H, G_rho_per_sigma = cmf.analysis.linear_fit_get_H(orbit_params["t"]/ketjugw.units.yr, orbit_params["a_R"]/ketjugw.units.pc, r_hard_time*1e6, tspan*1e6, snap, r_infl, return_Gps=True)
e0 = np.median(orbit_params["e_t"][r_hard_time_idx:a_more_Xpc])
print("e0: {:.3f}".format(e0))
print("G*rho/sigma: {:.3e}".format(G_rho_per_sigma))
print("Hardening rate: {:.4f}".format(H))
K = cmf.analysis.linear_fit_get_K(orbit_params["t"]/ketjugw.units.yr, orbit_params["e_t"], r_hard_time*1e6, tspan*1e6, H, G_rho_per_sigma, orbit_params["a_R"]/ketjugw.units.pc)
print("Eccentricity rate K: {:.4f}".format(K))

a_gr, a_gr_time = cmf.analysis.gravitational_radiation_radius(snap, r_infl, r_hard, r_hard_time, H, e=e0)
pq_estimates = TimeEstimates(orbit_params, r_hard_time_idx, a_more_Xpc, G_rho_per_sigma, H, K)

print("Hardening radius a_h: {:.2e}".format(r_hard))
print("GR emission radius a_GR: {:.2e}".format(a_gr))

#plotting
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios":[3,1]})
cmf.plotting.binary_param_plot(orbit_params, ax, toffset=time_offset, zorder=5)
ax[0].scatter(r_infl_time+time_offset, r_infl, zorder=10, label=r"$r_\mathrm{inf}$")
sc = ax[0].scatter(r_hard_time+time_offset, r_hard, zorder=10, label=r"$a_\mathrm{h}$")
ax[0].axvline(r_hard_time+tspan+time_offset, c="tab:red", label="H calculation")
ax[0].scatter(a_gr_time+time_offset, a_gr, zorder=10, label=r"$a_\mathrm{GR}$")
pq_estimates.plot(*ax, toffset=time_offset)
xaxis_lims = ax[0].get_xlim()
# TODO set a cosmologically-dependent Hubble time?
if xaxis_lims[1] > 13800:
    for axi in ax:
        axi.axvspan(13800, 1.1*xaxis_lims[1], color="k", alpha=0.4)
ax[0].set_xlim(*xaxis_lims)
ax[0].legend(loc="upper right")
plt.show()