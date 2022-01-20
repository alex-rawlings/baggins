from functools import cached_property
import os.path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import pygad

from .orbit import get_bound_binary, linear_fit_get_H, linear_fit_get_K, analytic_evolve_peters_quinlan
from .analyse_snap import *
from .general import snap_num_for_time, beta_profile
from .voronoi import voronoi_binned_los_V_statistics
from ..general import xval_of_quantity
from ..mathematics import spherical_components
from ..plotting import binary_param_plot
from ..utils import read_parameters, get_ketjubhs_in_dir, get_snapshots_in_dir

__all__ = ["BHBinary"]

myr = ketjugw.units.yr * 1e6


class BHBinary:
    def __init__(self, paramfile, perturbID, gr_safe_radius=15, time_estimate_quantiles=[0.05, 0.5, 0.95]):
        """
        A class to hold key quantities pertaining to the BH Binary. 
        cached_property method is used to prevent redoing expensive calculations

        Parameters
        ----------
        paramfile: parameter file with details about the parent merger, also 
                   file locations
        perturbID: number of the child run, to be used in file search
        gr_safe_radius: determine the analytical fits for when stellar 
                        scattering is the dominant hardening mechanism, which
                        is greater than this radius
        time_estimate_quantiles: estimate the merger time scales from these
                                 eccentricity quantile values
        """
        pfv = read_parameters(paramfile)
        data_path = os.path.join(pfv.full_save_location, pfv.perturbSubDir, perturbID, "output")
        self.bhfile = get_ketjubhs_in_dir(data_path)[0]
        self.snaplist = get_snapshots_in_dir(data_path)
        self.gr_safe_radius = gr_safe_radius
        bh1, bh2, self.merged = get_bound_binary(self.bhfile)
        self.orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)
        self.time_offset = pfv.perturbTime * 1e3
        self.time_estimate_quantiles = time_estimate_quantiles
    
    @cached_property
    def bh_masses(self):
        snap = pygad.Snapshot(self.snaplist[0], physical=True)
        return snap.bh["mass"]
    
    @cached_property
    def r_infl(self):
        snap = pygad.Snapshot(self.snaplist[0], physical=True)
        return list(influence_radius(snap).values())[0].in_units_of("pc") #in pc
    
    @property
    def r_hard(self):
        return hardening_radius(self.bh_masses, self.r_infl)
    
    def _get_time_for_r(self, r, desc="r"):
        try:
            return xval_of_quantity(r, self.orbit_params["t"]/myr, self.orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
        except ValueError:
            #when the radius is reached is not covered by the data
            if r > self.orbit_params["a_R"][0] / ketjugw.units.pc:
                warnings.warn("Time of {} cannot be estimated from the data, as it occurs before the binary is bound! This point will be omitted from further analysis and plots.".format(desc))
            elif r < self.orbit_params["a_R"][-1] / ketjugw.units.pc:
                warnings.warn("Time of {} cannot be estimated from the data, as it occurs after the last available data point! This point will be omitted from further analysis and plots.".format(desc))
            else:
                raise ValueError("{} cannot be determined".format(desc))
            return np.nan
    
    def _get_idx_in_vec(self, t, tarr):
        if not np.isnan(t):
            return np.argmax(t < tarr)
        else:
            return None
    
    @cached_property
    def r_infl_time(self):
        return self._get_time_for_r(self.r_infl, "r_infl")
    
    @cached_property
    def r_infl_time_idx(self):
        return self._get_idx_in_vec(self.r_infl_time, self.orbit_params["t"]/myr)
    
    @cached_property
    def r_hard_time(self):
        return self._get_time_for_r(self.r_hard, "r_hard")
    
    @cached_property
    def r_hard_time_idx(self):
        return self._get_idx_in_vec(self.r_hard_time, self.orbit_params["t"]/myr)

    @cached_property
    def a_more_Xpc_idx(self):
        idx = np.argmax(self.orbit_params["a_R"]/ketjugw.units.pc<self.gr_safe_radius)
        if idx < 2:
            idx = -1
        return idx
    
    @cached_property
    def tspan(self):
        return self.orbit_params["t"][self.a_more_Xpc_idx]/myr - self.r_hard_time
    
    @cached_property
    def G_rho_per_sigma(self):
        return get_G_rho_per_sigma(self.snaplist, self.r_hard_time, extent=self.r_infl)
    
    @cached_property
    def H(self):
        return linear_fit_get_H(self.orbit_params["t"]/ketjugw.units.yr, self.orbit_params["a_R"]/ketjugw.units.pc, self.r_hard_time*1e6, self.tspan*1e6, self.G_rho_per_sigma)
    
    @cached_property
    def K(self):
        return linear_fit_get_K(self.orbit_params["t"]/ketjugw.units.yr, self.orbit_params["e_t"], self.r_hard_time*1e6, self.tspan*1e6, self.H, self.G_rho_per_sigma, self.orbit_params["a_R"]/ketjugw.units.pc)
    
    @cached_property
    def gw_dominant_semimajoraxis(self):
        e0 = np.median(self.orbit_params["e_t"][self.r_hard_time_idx:self.a_more_Xpc_idx])
        a_gr, t_agr = gravitational_radiation_radius(
                                        self.bh_masses, self.r_hard, 
                                        self.r_hard_time, self.H,
                                        self.G_rho_per_sigma, e=e0)
        return {"a": a_gr, "t": t_agr}
    
    @property
    def time_estimate_quantiles(self):
        return self._time_estimate_quantiles
    
    @time_estimate_quantiles.setter
    def time_estimate_quantiles(self, q):
        assert len(q) == 3, "An upper, middle, and lower quantile must be specified"
        q.sort()
        self._time_estimate_quantiles = q
    
    @cached_property
    def predicted_orbital_params(self, idxs=None):
        op = dict(
            t = {"median":None, "upper":None, "lower":None},
            a = {"median":None, "upper":None, "lower":None},
            e = {"median":None, "upper":None, "lower":None}
        )
        if idxs is None:
            idxs = [self.r_hard_time_idx, self.a_more_Xpc_idx]
        assert isinstance(idxs, list)
        if idxs[1] != -1: assert idxs[0] < idxs[1]
        for k, q in zip(
                        ("lower", "median", "upper"), 
                        self.time_estimate_quantiles
                        ):
            a0 = self.orbit_params["a_R"][idxs[0]]
            e0 = np.nanquantile(self.orbit_params["e_t"][idxs[0]:idxs[1]], q)
            op["t"][k], op["a"][k], op["e"][k] = analytic_evolve_peters_quinlan(a0, e0, self.orbit_params["t"][idxs[0]], self.orbit_params["t"][-1], self.orbit_params["m0"][0], self.orbit_params["m1"][0], self.G_rho_per_sigma, self.H, self.K)
            op["t"][k] /= myr
            op["a"][k] /= ketjugw.units.pc
        return op
    
    @property
    def formation_eccentricity_spread(self):
        return np.nanstd(self.orbit_params["e_t"][:self.r_infl_time_idx])
    
    def plot(self, ax=None, add_radii=True, add_time_estimates=True, **kwargs):
        hubble_time = 13800
        if ax is None:
            fig, ax = plt.subplots(2,1,sharex="all")
        binary_param_plot(self.orbit_params, ax=ax, toffset=self.time_offset, zorder=5)
        if add_radii:
            ax[0].scatter(self.r_infl_time+self.time_offset, self.r_infl, zorder=10, label=r"$r_\mathrm{inf}$")
            ax[0].scatter(self.r_hard_time+self.time_offset, self.r_hard, zorder=10, label=r"$a_\mathrm{h}$")
            ax[0].axvspan(self.r_hard_time+self.time_offset, self.r_hard_time+self.tspan+self.time_offset, color="tab:red", alpha=0.4, label="H calculation")
            sc = ax[0].scatter(self.gw_dominant_semimajoraxis["t"]+self.time_offset, self.gw_dominant_semimajoraxis["a"], zorder=10, label=r"$a_\mathrm{GR}$")
            ax[0].axhline(self.gw_dominant_semimajoraxis["a"], ls=":", c=sc.get_facecolors())
        if add_time_estimates:
            for i, (k,q) in enumerate(
                            zip(self.predicted_orbital_params["a"].keys(), 
                            self.time_estimate_quantiles)):
                if i==0:
                    l = ax[0].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["a"][k], ls="--", label="{:.2f} quantile".format(q), **kwargs)
                else:
                    ax[0].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["a"][k], ls=":", c=l[-1].get_color(), label="{:.2f} quantile".format(q), **kwargs)
                ax[1].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["e"][k], ls=("--" if i%3==0 else ":"), c=l[-1].get_color(), **kwargs)
        xaxis_lims = ax[0].get_xlim()
        if xaxis_lims[1] > hubble_time:
            for axi in ax:
                axi.axvspan(hubble_time, 1.1*xaxis_lims[1], color="k", alpha=0.4)
        ax[0].set_xlim(*xaxis_lims)
        ax[0].legend(loc="upper right")
        return ax
    
    def print(self):
        print("BH Binary Quantities:")
        print("  Perturbation applied at {:.1f} Myr".format(self.time_offset))
        print("  H determined over a span of {:.1f} Myr".format(self.tspan))
        print("  G*rho/sigma: {:.3e}".format(self.G_rho_per_sigma))
        print("  Hardening rate H: {:.4f}".format(self.H))
        print("  Eccentricity rate K: {:.4f}".format(self.K))
        print("  Eccentricity scatter before r_infl: {:.2e}".format(self.formation_eccentricity_spread))
        if self.merged():
            print(self.merged)



class ChildDataCube(BHBinary):
    def __init__(self, paramfile, perturbID, gr_safe_radius=15, time_estimate_quantiles=[0.05, 0.5, 0.95], voronoi_kw={"Npx":300, "part_per_bin":5000}):
        """
        A class to hold information about a ketju child run, inheriting the
        attributes of the BHBinary class. 
        """
        super().__init__(paramfile, perturbID, gr_safe_radius, time_estimate_quantiles)
        self.voronoi_kw = voronoi_kw
        # TODO update file locations if necessary
    
    @cached_property
    def snap(self):
        merged_idx = snap_num_for_time(self.snaplist, self.merged.time, method="nearest")
        snap = pygad.Snapshot(self.snaplist[merged_idx], physical=True)
        xcom = get_com_of_each_galaxy(snap)
        vcom = get_com_velocity_of_each_galaxy(snap, xcom)
        snap["pos"] -= list(xcom.values())[0]
        snap["vel"] -= list(vcom.values())[0]
        return snap
    
    @cached_property
    def mass_centre(self):
        centre_guess = pygad.analysis.center_of_mass(self.snap.bh)
        return pygad.analysis.shrinking_sphere(self.snap.stars, centre_guess, 20)

    @property
    def binary_formation_time(self):
        return self.orbit_params["t"][0] / myr
    
    @property
    def binary_merger_timescale(self):
        if self.merged():
            return (self.orbit_params["t"][-1] - self.orbit_params["t"][0]) / myr
        else:
            return np.nan
    
    @property
    def binary_kick_velocity(self):
        if self.merged():
            return self.merged.kick_magnitude
        else:
            return np.nan

    @property
    def binary_spin_flip(self):
        return None

    @property
    def relaxed_stellar_velocity_dispersion(self):
        return np.nanstd(self.snap.stars["vel"], axis=0)
    
    @property
    def relaxed_stellar_velocity_dispersion_projected(self):
        #the property is returned from a function call that is part of another
        #property: call this other property first
        if not hasattr(self, "relaxed_effective_radius"):
            self.relaxed_effective_radius
        return self._vsig
    
    @cached_property
    def relaxed_inner_DM_fraction(self):
        return inner_DM_fraction(self.snap)

    @property
    def relaxed_core_size(self):
        return None
    
    @cached_property
    def relaxed_effective_radius(self):
        Re, self._vsig = projected_quantities(self.snap)
        return Re

    @cached_property
    def relaxed_half_mass_radius(self):
        return pygad.analysis.half_mass_radius(self.snap.stars, center=self.mass_centre)
    
    @property
    def relaxed_density_profile(self):
        return pygad.analysis.profile_dens(self.snap.stars, "mass", center=self.mass_centre)

    @property
    def relaxed_density_profile_projected(self):
        return None
    
    @property
    def relaxed_triaxiality_parameters(self):
        return None

    @property
    def total_stellar_mass(self):
        return self.snap.stars["mass"][0] * len(self.snap.stars)
    
    @property
    def ifu_map_ah(self):
        hard_idx = snap_num_for_time(self.snaplist, self.r_hard_time)
        snap = pygad.Snapshot(self.snaplist[hard_idx], physical=True)
        mass_centre = pygad.analysis.center_of_mass(snap.bh)
        # TODO is the best radius to use?
        ball_mask = pygad.BallMask(self.relaxed_effective_radius, center=mass_centre)
        subsnap = snap.stars[ball_mask]
        return voronoi_binned_los_V_statistics(
                    subsnap["pos"][:,0], subsnap["pos"][:,1], 
                    subsnap["vel"][:,2], subsnap["mass"], **self.voronoi_kw)
    
    @cached_property
    def ifu_map_merger(self):
        ball_mask = pygad.BallMask(self.relaxed_effective_radius, center=self.mass_centre)
        subsnap = self.snap.stars[ball_mask]
        return voronoi_binned_los_V_statistics(
                    subsnap["pos"][:,0], subsnap["pos"][:,1], 
                    subsnap["vel"][:,2], subsnap["mass"], **self.voronoi_kw)
    
    @property
    def stellar_shell_velocities(self):
        return None
    
    @property
    def beta_r(self):
        vspherical = spherical_components(self.snap.stars["pos"], self.snap.star["vel"])
        beta_r, radbins, bincount = beta_profile(self.snap.stars["r"], vspherical, 0.05)
        vals = {"beta_r": beta_r, "radbins":radbins, "bincount": bincount}
        return vals
    
    def make_hdf5(self):
        raise NotImplementedError
    
    def plot(self):
        warnings.warn("This method is not available for this class. If you wish to plot the BH Binary orbital parameters, use the binary_plot() method.")
    
    def binary_plot(self, ax=None, add_radii=True, add_time_estimates=True, **kwargs):
        super().plot(ax, add_radii, add_time_estimates, **kwargs)

    def print(self):
        #print the currently loaded attributes of the data cube
        s = "Data Cube has the following properties loaded:\n"
        keys = self.__dict__.keys()
        for k in keys:
            if k[0] != "_":
                s += " > {}\n".format(k)
        print(s)

