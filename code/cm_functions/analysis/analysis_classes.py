import os.path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import pygad

from .orbit import get_bound_binary, linear_fit_get_H, linear_fit_get_K, analytic_evolve_peters_quinlan
from .analyse_snap import influence_radius, hardening_radius, gravitational_radiation_radius, get_G_rho_per_sigma
from .general import snap_num_for_time
from ..general import xval_of_quantity
from ..plotting import binary_param_plot
from ..utils import read_parameters, get_ketjubhs_in_dir, get_snapshots_in_dir

__all__ = ["BHBinary"]

myr = ketjugw.units.yr * 1e6


class BHBinary:
    def __init__(self, paramfile, perturbID, gr_safe_radius=15):
        pfv = read_parameters(paramfile)
        data_path = os.path.join(pfv.full_save_location, pfv.perturbSubDir, perturbID, "output")
        self.bhfile = get_ketjubhs_in_dir(data_path)[0]
        self.snaplist = get_snapshots_in_dir(data_path)
        self.gr_safe_radius = gr_safe_radius
        bh1, bh2, self.merged = get_bound_binary(self.bhfile)
        self.orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)
        self.time_offset = pfv.perturbTime * 1e3
        self.has_characteristic_radii = False
        self.bh_masses = None
        self.r_infl = None
        self.r_infl_time = None
        self.r_hard = None
        self.r_hard_time = None
        self.r_hard_time_idx = None
        self.r_infl_time_idx = None
        self.hard_snap_idx = None
        self.has_analytic_estimate = False
        self.tspan = None
        self.a_more_Xpc_idx = None
        self.H = None
        self.G_rho_per_sigma = None
        self.K = None
        self.a_gr = None
        self.a_gr_time = None
        self.has_time_estimates = False
        self.quantiles = None
        self.predict_a = None
        self.predict_e = None
        self.predict_t = None
        self.e0_spread = None

        print("Perturbation applied at {:.1f} Myr".format(self.time_offset))

    def get_influence_and_hard_radius(self):
        self.has_characteristic_radii = True
        myr = ketjugw.units.yr * 1e6
        snap = pygad.Snapshot(self.snaplist[0], physical=True)
        self.bh_masses = snap.bh["mass"]
        self.r_infl = list(influence_radius(snap).values())[0].in_units_of("pc") #in pc
        self.r_hard = hardening_radius(snap.bh["mass"], self.r_infl)
        #determine when the above radii occur
        #and the corresponding indices
        try:
            self.r_infl_time = xval_of_quantity(self.r_infl, self.orbit_params["t"]/myr, self.orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
            self.r_infl_time_idx = np.argmax(
                            self.r_infl_time < self.orbit_params["t"]/myr)
        except ValueError:
            #when the influence radius is reached is not covered by the data
            if self.r_infl > self.orbit_params["a_R"][0] / ketjugw.units.pc:
                warnings.warn("Time of r_infl cannot be estimated from the data, as it occurs before the binary is bound! This point will be omitted from further analysis and plots.")
            elif self.r_infl < self.orbit_params["a_R"][-1] / ketjugw.units.pc:
                warnings.warn("Time of r_infl cannot be estimated from the data, as it occurs after the last available data point! This point will be omitted from further analysis and plots.")
            else:
                raise ValueError("r_infl cannot be determined")
            self.r_infl_time = np.nan
        self.r_hard_time = xval_of_quantity(self.r_hard, self.orbit_params["t"]/myr, self.orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
        self.r_hard_time_idx = np.argmax(
                            self.r_hard_time < self.orbit_params["t"]/myr)
    
    def fit_analytic_form(self):
        self.has_analytic_estimate = True
        #get the snap from which inner density, sigma is found
        #snap = pygad.Snapshot(self.snaplist[self.hard_snap_idx], physical=True)
        self.a_more_Xpc_idx = np.argmax(
            self.orbit_params["a_R"]/ketjugw.units.pc<self.gr_safe_radius)
        if self.a_more_Xpc_idx < 2:
            self.a_more_Xpc_idx = -1
        time_a_more_Xpc = self.orbit_params["t"][self.a_more_Xpc_idx]/myr
        self.tspan = time_a_more_Xpc - self.r_hard_time
        print("H determined over a span of {:.3f} Myr".format(self.tspan))

        #determine hardening constants -- times are in years
        self.G_rho_per_sigma = get_G_rho_per_sigma(self.snaplist, self.r_hard_time, extent=self.r_infl)
        self.H = linear_fit_get_H(self.orbit_params["t"]/ketjugw.units.yr, self.orbit_params["a_R"]/ketjugw.units.pc, self.r_hard_time*1e6, self.tspan*1e6, self.G_rho_per_sigma)
        e0 = np.median(
            self.orbit_params["e_t"][self.r_hard_time_idx:self.a_more_Xpc_idx])
        print("e0: {:.3f}".format(e0))
        print("G*rho/sigma: {:.3e}".format(self.G_rho_per_sigma))
        print("Hardening rate: {:.4f}".format(self.H))
        self.K = linear_fit_get_K(self.orbit_params["t"]/ketjugw.units.yr, self.orbit_params["e_t"], self.r_hard_time*1e6, self.tspan*1e6, self.H, self.G_rho_per_sigma, self.orbit_params["a_R"]/ketjugw.units.pc)
        print("Eccentricity rate K: {:.4f}".format(self.K))

        self.a_gr, self.a_gr_time = gravitational_radiation_radius(
                                        self.bh_masses, self.r_hard, 
                                        self.r_hard_time, self.H,
                                        self.G_rho_per_sigma, e=e0)
    
    def time_estimates(self, idxs=None, quantiles=(0.5, 0.05, 0.95)):
        self.quantiles = quantiles
        assert len(self.quantiles) == 3, "An upper, middle, and lower quantile must be specified"
        self.has_time_estimates = True
        self.predict_a = {"median":None, "upper":None, "lower":None}
        self.predict_e = {"median":None, "upper":None, "lower":None}
        self.predict_t = {"median":None, "upper":None, "lower":None}
        if idxs is None:
            idxs = [self.r_hard_time_idx, self.a_more_Xpc_idx]
        assert isinstance(idxs, list)
        if idxs[1] != -1: assert idxs[0] < idxs[1]

        for k, q in zip(("median", "upper", "lower"), self.quantiles):
            a0 = self.orbit_params["a_R"][idxs[0]]
            e0 = np.nanquantile(self.orbit_params["e_t"][idxs[0]:idxs[1]], q)
            self.predict_t[k], self.predict_a[k], self.predict_e[k] = analytic_evolve_peters_quinlan(a0, e0, 
                self.orbit_params["t"][idxs[0]], self.orbit_params["t"][-1], 
                self.orbit_params["m0"][0], self.orbit_params["m1"][0], 
                self.G_rho_per_sigma, self.H, self.K)
    
    def formation_eccentricity_spread(self):
        self.e0_spread = np.nanstd(self.orbit_params["e_t"][:self.r_infl_time_idx])

    def plot(self, ax=None, toffset=None, **kwargs):
        if toffset is None:
            toffset = self.time_offset
        myr = ketjugw.units.yr * 1e6
        if ax is None:
            fig, ax = plt.subplots(2,1,sharex="all")
        binary_param_plot(self.orbit_params, ax=ax, toffset=toffset, zorder=5)
        if self.has_characteristic_radii:
            ax[0].scatter(self.r_infl_time+toffset, self.r_infl, zorder=10, label=r"$r_\mathrm{inf}$")
            ax[0].scatter(self.r_hard_time+toffset, self.r_hard, zorder=10, label=r"$a_\mathrm{h}$")
            ax[0].axvspan(self.r_hard_time+toffset, self.r_hard_time+self.tspan+toffset, color="tab:red", alpha=0.4, label="H calculation")
            sc = ax[0].scatter(self.a_gr_time+toffset, self.a_gr, zorder=10, label=r"$a_\mathrm{GR}$")
            ax[0].axhline(self.a_gr, ls=":", c=sc.get_facecolors())
        if self.has_time_estimates:
            for i, (k,q) in enumerate(
                            zip(self.predict_a.keys(), self.quantiles)):
                if i==0:
                    l = ax[0].plot(self.predict_t[k]/myr+toffset, self.predict_a[k]/ketjugw.units.pc, ls="--", label="{:.2f} quantile".format(q), **kwargs)
                else:
                    ax[0].plot(self.predict_t[k]/myr+toffset, self.predict_a[k]/ketjugw.units.pc, ls=":", c=l[-1].get_color(), label="{:.2f} quantile".format(q), **kwargs)
                ax[1].plot(self.predict_t[k]/myr+toffset, self.predict_e[k], ls=("--" if i%3==0 else ":"), c=l[-1].get_color(), **kwargs)
        xaxis_lims = ax[0].get_xlim()
        if xaxis_lims[1] > 13800:
            for axi in ax:
                axi.axvspan(13800, 1.1*xaxis_lims[1], color="k", alpha=0.4)
        ax[0].set_xlim(*xaxis_lims)
        ax[0].legend(loc="upper right")
        return ax


class ChildDataCube(BHBinary):
    def __init__(self, paramfile, perturbID, gr_safe_radius=15):
        super().__init__(paramfile, perturbID, gr_safe_radius)
        self.binary_formation_time = None
        self.binary_merger_timescale = None
        self.binary_spin_flip = None
        self.binary_kick_velocity = None
        self.relaxed_stellar_velocity_dispersion = None
        self.relaxed_inner_DM_fraction = None
        self.relaxed_core_size = None
        self.relaxed_effective_radius = None
        self.relaxed_half_mass_radius = None
        self.relaxed_density_profile = None
        self.relaxed_density_profile_projected = None
        self.total_stellar_mass = None
        self.ifu_map_ah = None
        self.ifu_map_merger = None
        self.stellar_shell_velocities = None
    
    def find_binary_timescales(self):
        self.binary_formation_time = self.orbit_params["t"][0] / myr
        if self.merged:
            self.binary_merger_timescale = (self.orbit_params["t"][-1] - self.orbit_params["t"][0]) / myr
        else:
            self.binary_merger_timescale = np.nan
    
    def find_binary_flip_and_kick(self):
        pass

