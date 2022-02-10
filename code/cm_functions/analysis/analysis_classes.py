import datetime
from functools import cached_property
import os.path
import warnings
import numpy as np
import h5py
import matplotlib.pyplot as plt
import ketjugw
import pygad

from .orbit import get_bound_binary, linear_fit_get_H, linear_fit_get_K, analytic_evolve_peters_quinlan
from .analyse_snap import *
from .general import snap_num_for_time, beta_profile
from .voronoi import voronoi_binned_los_V_statistics
from ..general import xval_of_quantity, convert_gadget_time
from ..mathematics import spherical_components
from ..plotting import binary_param_plot
from ..utils import read_parameters, get_ketjubhs_in_dir, get_snapshots_in_dir

__all__ = ["BHBinary", "ChildDataCube"]

myr = ketjugw.units.yr * 1e6
date_str = "%Y-%m-%d %H:%M:%S"


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

    def predict_gw_dominant_semimajoraxis(self, q):
        e0 = np.nanquantile(self.orbit_params["e_t"][self.r_hard_time_idx:self.a_more_Xpc_idx], q)
        a_gr, t_agr = gravitational_radiation_radius(
                                        self.bh_masses, self.r_hard, 
                                        self.r_hard_time, self.H,
                                        self.G_rho_per_sigma, e=e0)
        return a_gr, t_agr
    
    @cached_property
    def gw_dominant_semimajoraxis(self):
        a_gr, t_agr = self.predict_gw_dominant_semimajoraxis(0.5)
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
            t = {"lower":None, "median":None, "upper":None},
            a = {"lower":None, "median":None, "upper":None},
            e = {"lower":None, "median":None, "upper":None}
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
            agr_low,_ = self.predict_gw_dominant_semimajoraxis(self.time_estimate_quantiles[0])
            agr_high,_ = self.predict_gw_dominant_semimajoraxis(self.time_estimate_quantiles[-1])
            ax[0].axhspan(agr_low, agr_high, facecolor=sc.get_facecolors(), alpha=0.4)
        if add_time_estimates:
            for i, (k,q) in enumerate(
                            zip(self.predicted_orbital_params["a"].keys(), 
                            self.time_estimate_quantiles)):
                if i==0:
                    l = ax[0].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["a"][k], ls=":", label="{:.2f} quantile".format(q), **kwargs)
                else:
                    ax[0].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["a"][k], ls=("--" if i==1 else ":"), c=l[-1].get_color(), label="{:.2f} quantile".format(q), **kwargs)
                ax[1].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["e"][k], ls=("--" if i%3==1 else ":"), c=l[-1].get_color(), **kwargs)
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
    def __init__(self, paramfile, perturbID, gr_safe_radius=15, time_estimate_quantiles=[0.05, 0.5, 0.95], voronoi_kw={"Npx":300, "part_per_bin":5000}, shell_radius=3e-2):
        """
        A class to hold information about a ketju child run, inheriting the
        attributes of the BHBinary class. 
        """
        super().__init__(paramfile, perturbID, gr_safe_radius, time_estimate_quantiles)
        self.voronoi_kw = voronoi_kw
        self.shell_radius = shell_radius
        # TODO update file locations if necessary
    
    @cached_property
    def snap(self):
        merged_idx = snap_num_for_time(self.snaplist, self.merged.time, method="nearest")
        snap = pygad.Snapshot(self.snaplist[merged_idx], physical=True)
        xcom = get_com_of_each_galaxy(snap, verbose=False)
        vcom = get_com_velocity_of_each_galaxy(snap, xcom, verbose=False)
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
        raise NotImplementedError

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
        raise NotImplementedError
    
    @cached_property
    def relaxed_effective_radius(self):
        Re, vsig = projected_quantities(self.snap)
        self._vsig = list(vsig.values())[0]
        return list(Re.values())[0]

    @cached_property
    def relaxed_half_mass_radius(self):
        return pygad.analysis.half_mass_radius(self.snap.stars, center=self.mass_centre)
    
    @property
    def relaxed_density_profile(self):
        return pygad.analysis.profile_dens(self.snap.stars, "mass", center=self.mass_centre)

    @property
    def relaxed_density_profile_projected(self):
        raise NotImplementedError
    
    @property
    def relaxed_triaxiality_parameters(self):
        raise NotImplementedError

    @property
    def total_stellar_mass(self):
        return self.snap.stars["mass"][0] * len(self.snap.stars)
    
    @cached_property
    def ifu_map_ah(self):
        hard_idx = snap_num_for_time(self.snaplist, self.r_hard_time)
        snap = pygad.Snapshot(self.snaplist[hard_idx], physical=True)
        mass_centre = pygad.analysis.center_of_mass(snap.bh)
        # TODO is the best radius to use?
        ball_mask = pygad.BallMask(self.relaxed_effective_radius["estimate"], center=mass_centre)
        subsnap = snap.stars[ball_mask]
        return voronoi_binned_los_V_statistics(
                    subsnap["pos"][:,0], subsnap["pos"][:,1], 
                    subsnap["vel"][:,2], subsnap["mass"], **self.voronoi_kw)
    
    @cached_property
    def ifu_map_merger(self):
        ball_mask = pygad.BallMask(self.relaxed_effective_radius["estimate"], center=self.mass_centre)
        subsnap = self.snap.stars[ball_mask]
        return voronoi_binned_los_V_statistics(
                    subsnap["pos"][:,0], subsnap["pos"][:,1], 
                    subsnap["vel"][:,2], subsnap["mass"], **self.voronoi_kw)
    
    @property
    def snapshot_times(self):
        if not hasattr(self, "_snapshot_times"):
            raise AttributeError("Must call get_shell_velocity_stats first!")
        return self._snapshot_times

    @property
    def stellar_shell_outflow_velocity(self):
        if not hasattr(self, "_snapshot_times"):
            raise AttributeError("Must call get_shell_velocity_stats first!")
        return self._stellar_shell_outflow_velocity
    
    @property
    def bh_binary_watershed_velocity(self):
        if not hasattr(self, "_snapshot_times"):
            raise AttributeError("Must call get_shell_velocity_stats first!")
        return self._bh_binary_watershed_velocity
    
    def get_shell_velocity_stats(self, R=3e-2):
        bound_snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][0]/myr)
        last_snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][-1]/myr)
        self._snapshot_times = np.full(last_snap_idx-bound_snap_idx+1, np.nan)
        self._bh_binary_watershed_velocity = np.full(last_snap_idx - bound_snap_idx + 1, np.nan)
        self._stellar_shell_outflow_velocity = []
        m_min = min([self.orbit_params["m0"][0], self.orbit_params["m1"][0]])
        for i, j in enumerate(np.arange(bound_snap_idx, last_snap_idx+1)):
            snap = pygad.Snapshot(self.snaplist[j], physical=True)
            #as we are interested in flow rates about binary, set binary as
            #the centre of mass
            this_centre = pygad.analysis.center_of_mass(snap.bh)
            self._snapshot_times[i] = convert_gadget_time(snap, new_unit="Myr")
            idx = self._get_idx_in_vec(self._snapshot_times[i], self.orbit_params["t"]/myr)
            self._bh_binary_watershed_velocity[i] = 0.85 * np.sqrt(m_min / self.orbit_params["a_R"][idx]) / ketjugw.units.km_per_s
            self._stellar_shell_outflow_velocity.append(
                shell_flow_velocities(snap.stars, R, centre=this_centre, direction="in")
            )
            snap.delete_blocks()
    
    @cached_property
    def beta_r(self):
        vspherical = spherical_components(self.snap.stars["pos"], self.snap.stars["vel"])
        beta_r, radbins, bincount = beta_profile(self.snap.stars["r"], vspherical, 0.05)
        return {"beta_r": beta_r, "radbins":radbins, "bincount": bincount}
    
    @cached_property
    def virial_info(self):
        v_rad, v_mass = pygad.analysis.virial_info(self.snap, center=self.mass_centre)
        return {"virial_radius":v_rad, "virial_mass":v_mass}
    
    def load_all(self, verbose=True):
        if verbose: print("Loading BH masses")
        self.bh_masses
        if verbose: print("Loading influence radius, hardening radius, GW radius, and their associated time estimates")
        self.r_infl
        self.r_hard
        self.r_infl_time
        self.r_hard_time
        self.gw_dominant_semimajoraxis
        if verbose: print("Loading analytical forms")
        self.G_rho_per_sigma
        self.H
        self.K
        self.mass_centre
        if verbose: print("Loading binary timescales")
        self.binary_formation_time
        self.binary_merger_timescale
        if verbose: print("Loading kick velocity")
        self.binary_kick_velocity
        if verbose: print("Loading stellar velocity dispersion")
        self.relaxed_stellar_velocity_dispersion
        self.relaxed_inner_DM_fraction
        if verbose: print("Loading half-mass radius")
        self.relaxed_half_mass_radius
        if verbose: print("Loading density profile")
        self.relaxed_density_profile
        self.total_stellar_mass
        if verbose: print("Loading IFU maps")
        self.ifu_map_ah
        self.ifu_map_merger
        if verbose: print("Loading shell velocity statistics")
        self.get_shell_velocity_stats()
        if verbose: print("Loading beta profile")
        self.beta_r
        if verbose: print("Loading virial info")
        self.virial_info
        if verbose: print("All attributes loaded")


    def make_hdf5(self, fname):
        # TODO: here we are assuming all attributes are either loaded or can
        # be existed... should we include a test so that only those attributes
        # which have been explicitly determined be written?
        with h5py.File(fname, mode="w") as f:
            #set up some meta data
            meta = f.create_group("meta")
            now = datetime.datetime.now()
            meta.attrs["created"] = now.strftime(date_str)
            usr = os.path.expanduser("~")
            meta.attrs["created_by"] = usr.rstrip("/").split("/")[-1]
            # TODO move this to the read_hdf5 method
            now = datetime.datetime.now()
            meta.attrs["last_accessed"] = now.strftime(date_str)
            usr = os.path.expanduser("~")
            meta.attrs["last_user"] = usr.rstrip("/").split("/")[-1]
            
            #save the binary info
            bhb = f.create_group("bh_binary")
            bhb.create_dataset("binary_formation_time", data=self.binary_formation_time)
            bhb.create_dataset("binary_merger_timescale", data=self.binary_merger_timescale)
            bhb_m = bhb.create_group("merger_remnant")
            bhb_m.create_dataset("merged", data=self.merged.merged)
            bhb_m.create_dataset("mass", data=self.merged.mass)
            bhb_m.create_dataset("kick_magnitude", data=self.merged.kick_magnitude)
            bhb_m.create_dataset("chi", data=self.merged.chi)
            bhb.create_dataset("r_infl", data=self.r_infl)
            bhb.create_dataset("r_infl_time", data=self.r_infl_time)
            bhb.create_dataset("r_hard", data=self.r_hard)
            bhb.create_dataset("r_hard_time", data=self.r_hard_time)
            bhb.create_dataset("tspan", data=self.tspan)
            bhb.create_dataset("H", data=self.H)
            bhb.create_dataset("K", data=self.K)
            bhb.create_dataset("gw_dominant_semimajoraxis", data=self.gw_dominant_semimajoraxis["a"])
            bhb.create_dataset("gw_dominant_semimajoraxis_time", data=self.gw_dominant_semimajoraxis["t"])
            bhb_op = bhb.create_group("predicted_orbital_params")
            bhb_op_t = bhb_op.create_group("time")
            bhb_op_a = bhb_op.create_group("a")
            bhb_op_e = bhb_op.create_group("e")
            for k in self.predicted_orbital_params["a"].keys():
                bhb_op_t.create_dataset(k, data=self.predicted_orbital_params["t"][k])
                bhb_op_a.create_dataset(k, data=self.predicted_orbital_params["a"][k])
                bhb_op_e.create_dataset(k, data=self.predicted_orbital_params["e"][k])
            bhb.create_dataset("formation_eccentricity_spread", data=self.formation_eccentricity_spread)

            #save the galaxy properties
            gp = f.create_group("galaxy_properties")
            gp.create_dataset("stellar_velocity_dispersion", data=self.relaxed_stellar_velocity_dispersion)
            gp_sp = gp.create_group("stellar_velocity_dispersion_projected")
            for k,v in self.relaxed_stellar_velocity_dispersion_projected.items():
                gp_sp.create_dataset(k, data=v)
            gp.create_dataset("inner_DM_fraction", data=self.relaxed_inner_DM_fraction)
            gp.create_dataset("half_mass_radius", data=self.relaxed_half_mass_radius)
            gp_er = gp.create_group("effective_radius")
            for k,v in self.relaxed_effective_radius.items():
                gp_er.create_dataset(k, data=v)
            gp_dens = gp.create_group("density_profile")
            gp_dens.create_dataset("3D", data=self.relaxed_density_profile)
            #gp_dens.create_dataset("projected", data=self.relaxed_density_profile_projected)
            gp.create_dataset("total_stellar_mass", data=self.total_stellar_mass)
            gp_ifu_m = gp.create_group("ifu_map_merger")
            for k,v in self.ifu_map_merger.items():
                gp_ifu_m.create_dataset(k, data=v)
            gp_ifu_a = gp.create_group("ifu_map_ah")
            for k,v in self.ifu_map_ah.items():
                gp_ifu_a.create_dataset(k, data=v)
            gp_shell = gp.create_group("shell_statistics")
            gp_shell.attrs["shell_radius"] = self.shell_radius
            gp_shell.create_dataset("snapshot_times", data=self.snapshot_times)
            gp_shell.create_dataset("stellar_shell_outflow_velocity", data=self.stellar_shell_outflow_velocity)
            gp_shell.create_dataset("bh_binary_watershed_velocity", data=self.bh_binary_watershed_velocity)
            gp_beta = gp.create_group("beta_r")
            for k,v in self.beta_r.items():
                gp_beta.create_dataset(k, data=v)
            gp_vir = gp.create_group("virial_info")
            for k,v in self.virial_info.items():
                gp_vir.create_dataset(k, data=v)

    # TODO: option to save class as pickle so that methods can also be reloaded?

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
