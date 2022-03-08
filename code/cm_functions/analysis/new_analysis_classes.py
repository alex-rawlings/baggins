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
from .masks import get_radial_mask
from .voronoi import voronoi_binned_los_V_statistics
from ..general import xval_of_quantity, convert_gadget_time
from ..mathematics import spherical_components, iqr, get_histogram_bin_centres
from ..plotting import binary_param_plot
from ..utils import read_parameters, get_ketjubhs_in_dir, get_snapshots_in_dir
from ..literature import fit_Terzic05_profile, Terzic05
from ..env_config import username


myr = ketjugw.units.yr * 1e6
date_str = "%Y-%m-%d %H:%M:%S"
hubble_time = 13800


class BHBinaryData:
    def __init__(self) -> None:
        """
        A class that defines the fields which constitute the variables of 
        interest for the BH Binary. These properties are accessible to all 
        child classes, and also correspond to the fields which are loadable 
        from a hdf5 file. 
        """
        self._log = ""
    
    @property
    def snaplist(self):
        return self._snaplist
    
    @snaplist.setter
    def snaplist(self, v):
        bad_snaps = []
        for i, vi in enumerate(v):
            try:
                s = pygad.Snapshot(vi, physical=True)
                s.delete_blocks()
            except KeyError:
                msg = "Snapshot {} potentially corrupt. Removing it from the list of snapshots for further analysis!".format(vi)
                warnings.warn(msg)
                self.add_to_log(msg)
                bad_snaps.append(vi)
        self._snaplist = [x for x in v if x not in bad_snaps]

    @property
    def bh_masses(self):
        return self._bh_masses
    
    @bh_masses.setter
    def bh_masses(self, v):
        self._bh_masses = v
    
    @property
    def binary_formation_time(self):
        return self._binary_formation_time
    
    @binary_formation_time.setter
    def binary_formation_time(self, v):
        self._binary_formation_time = v
    
    @property
    def binary_merger_timescale(self):
        return self._binary_merger_timescale
    
    @binary_merger_timescale.setter
    def binary_merger_timescale(self, v):
        self._binary_merger_timescale = v
    
    @property
    def r_infl(self):
        return self._r_infl
    
    @r_infl.setter
    def r_infl(self, v):
        self._r_infl = v
    
    @property
    def r_infl_time(self):
        return self._r_infl_time
    
    @r_infl_time.setter
    def r_infl_time(self, v):
        self._r_infl_time = v
    
    @property
    def r_infl_time_idx(self):
        return self._r_infl_time_idx
    
    @r_infl_time_idx.setter
    def r_infl_time_idx(self, v):
        self._r_infl_time_idx = v
    
    @property
    def r_infl_ecc(self):
        return self._r_infl_ecc
    
    @r_infl_ecc.setter
    def r_infl_ecc(self, v):
        self._r_infl_ecc = v
    
    @property
    def r_bound(self):
        return self._r_bound
    
    @r_bound.setter
    def r_bound(self, v):
        self._r_bound = v
    
    @property
    def r_bound_time(self):
        return self._r_bound_time
    
    @r_bound_time.setter
    def r_bound_time(self, v):
        self._r_bound_time = v
    
    @property
    def r_bound_time_idx(self):
        return self._r_bound_time_idx
    
    @r_bound_time_idx.setter
    def r_bound_time_idx(self, v):
        self._r_bound_time_idx = v
    
    @property
    def r_bound_ecc(self):
        return self._r_bound_ecc
    
    @r_bound_ecc.setter
    def r_bound_ecc(self, v):
        self._r_bound_ecc = v

    @property
    def r_hard(self):
        return self._r_hard
    
    @r_hard.setter
    def r_hard(self, v):
        self._r_hard = v
    
    @property
    def r_hard_time(self):
        return self._r_hard_time
    
    @r_hard_time.setter
    def r_hard_time(self, v):
        self._r_hard_time = v
    
    @property
    def r_hard_time_idx(self):
        return self._r_hard_time_idx
    
    @r_hard_time_idx.setter
    def r_hard_time_idx(self, v):
        self._r_hard_time_idx = v
    
    @property
    def r_hard_ecc(self):
        return self._r_hard_ecc
    
    @r_hard_ecc.setter
    def r_hard_ecc(self, v):
        self._r_hard_ecc = v
    
    @property
    def a_more_Xpc_idx(self):
        return self._a_more_Xpc_idx
    
    @a_more_Xpc_idx.setter
    def a_more_Xpc_idx(self, v):
        self._a_more_Xpc_idx = v
    
    @property
    def analytical_tspan(self):
        return self._analytical_tspan
    
    @analytical_tspan.setter
    def analytical_tspan(self, v):
        self._analytical_tspan = v
    
    @property
    def G_rho_per_sigma(self):
        return self._G_rho_per_sigma
    
    @G_rho_per_sigma.setter
    def G_rho_per_sigma(self, v):
        self._G_rho_per_sigma = v

    @property
    def H(self):
        return self._H
    
    @H.setter
    def H(self, v):
        self._H = v
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, v):
        self._K = v
    
    @property
    def gw_dominant_semimajoraxis(self):
        return self._gw_dominant_semimajoraxis
    
    @gw_dominant_semimajoraxis.setter
    def gw_dominant_semimajoraxis(self, v):
        self._gw_dominant_semimajoraxis = v
    
    @property
    def param_estimate_e_quantiles(self):
        return self._param_estimate_e_quantiles
    
    @param_estimate_e_quantiles.setter
    def param_estimate_e_quantiles(self, v):
        assert len(v) == 3, "An upper, middle, and lower quantile must be specified"
        v.sort()
        self._param_estimate_e_quantiles = v
    
    @property
    def predicted_orbital_params(self):
        return self._predicted_orbital_params
    
    @predicted_orbital_params.setter
    def predicted_orbital_params(self, v):
        self._predicted_orbital_params = v
    
    @property
    def formation_ecc_spread(self):
        return self._formation_ecc_spread
    
    @formation_ecc_spread.setter
    def formation_ecc_spread(self, v):
        self._formation_ecc_spread = v
    
    @property
    def binary_merger_remnant(self):
        return self._binary_merger_remnant
    
    @binary_merger_remnant.setter
    def binary_merger_remnant(self, v):
        self._binary_merger_remnant = v
    
    @property
    def binary_spin_flip(self):
        return self._binary_spin_flip
    
    @binary_spin_flip.setter
    def binary_spin_flip(self, v):
        self._binary_spin_flip = v

    def add_to_log(self, msg):
        #add a message to the log
        now = datetime.datetime.now()
        now = now.strftime(date_str)
        self._log += (now+": "+msg+"\n")



class BHBinary(BHBinaryData):
    def __init__(self, paramfile, perturbID, gr_safe_radius=15, param_estimate_e_quantiles=[0.05, 0.5, 0.95]) -> None:
        """
        A class which determines and sets the key BH binary properties from the
        raw simulation output data. 

        Parameters
        ----------
        paramfile: path to parameter file corresponding to the merger run
        perturbID: name of the perturbation directory, e.g. 000
        gr_safe_radius: semimajor axis above which hardening due to GR emission
                        should be negligible [pc]
        param_estimate_e_quantiles: estimate binary orbital parameters assuming
                                    an initial eccentricity of these quantiles
                                    from a_h to gr_safe_radius
        """
        super().__init__()
        pfv = read_parameters(paramfile)
        data_path = os.path.join(pfv.full_save_location, pfv.perturbSubDir, perturbID, "output")
        if not os.path.isdir(data_path):
            raise ValueError("The data path does not exist!")
        self.merger_name = "{}-{}".format(pfv.full_save_location.rstrip("/").split("/")[-1], perturbID)
        self.bhfile = get_ketjubhs_in_dir(data_path)[0]
        self.snaplist = get_snapshots_in_dir(data_path)
        self.gr_safe_radius = gr_safe_radius
        bh1, bh2, self.merged = get_bound_binary(self.bhfile)
        self.orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)
        self.time_offset = pfv.perturbTime * 1e3
        self.param_estimate_e_quantiles = param_estimate_e_quantiles

        #set the properties
        #black hole mass
        snap = pygad.Snapshot(self.snaplist[0], physical=True)
        self.bh_masses = snap.bh["mass"]

        #time when binary is 'formed'
        self.binary_formation_time = self.time_offset + self.orbit_params["t"][0] / myr

        #merger timescale
        if self.merged:
            self.binary_merger_timescale = (self.orbit_params["t"][-1] - self.orbit_params["t"][0]) / myr
        else:
            self.binary_merger_timescale = np.nan
        
        #get the main properties of the binary remnant
        if self.merged():
            self.binary_merger_remnant = dict(
                merged = self.merged.merged,
                mass = self.merged.mass,
                chi = self.merged.chi,
                kick = self.merged.kick_magnitude
            )
        else:
            self.binary_merger_remnant = dict(
                merged = self.merged.merged,
                mass = np.nan,
                chi = np.nan,
                kick = np.nan
            )
        
        #set the spin flip bool, defined as Lz changing sign (Nasim 2021)
        self.binary_spin_flip = np.any(np.abs(np.diff(np.sign(self.orbit_params["plane_normal"][:,2])))>0)

        #influence radius
        self.r_infl = list(influence_radius(snap).values())[0].in_units_of("pc") #in pc

        #influence radius time
        self.r_infl_time = self._get_time_for_r(self.r_infl, "r_infl")

        try:
            #influence radius time index
            self.r_infl_time_idx = self._get_idx_in_vec(self.r_infl_time, self.orbit_params["t"]/myr)

            #eccentricity at time of influence radius
            self.r_infl_ecc = self._get_ecc_for_idx(self.r_infl_time_idx)

            #determine the spread in eccentricity at formation
            # TODO is IQR better than standard deviation?
            self.formation_ecc_spread = iqr(self.orbit_params["e_t"][:self.r_infl_time_idx])
        except ValueError:
            self.r_infl_ecc = np.nan
            self.formation_ecc_spread = np.nan

        #bound radius (Gualandris et al. 2022)
        self.r_bound = list(enclosed_mass_radius(snap, mass_frac=0.1).values())[0].in_units_of("pc")

        #bound radius time
        self.r_bound_time = self._get_time_for_r(self.r_bound, "r_bound")

        try:
            #bound radius time index
            self.r_bound_time_idx = self._get_idx_in_vec(self.r_bound_time, self.orbit_params["t"]/myr)

            #eccentricity at time of bound radius
            self.r_bound_ecc = self._get_ecc_for_idx(self.r_bound_time_idx)
        except ValueError:
            self.r_bound_ecc = np.nan

        #hardening radius
        self.r_hard = hardening_radius(self.bh_masses, self.r_infl)

        #hardening radius time
        self.r_hard_time = self._get_time_for_r(self.r_hard, "r_hard")

        try:
            #hardening radius time index
            self.r_hard_time_idx = self._get_idx_in_vec(self.r_hard_time, self.orbit_params["t"]/myr)

            #eccentricity at time of bound radius
            self.r_hard_ecc = self._get_ecc_for_idx(self.r_hard_time_idx)
        except ValueError("r_hard_time_idx not valid -> setting to 0"):
            self.r_hard_time_idx = 0

        #index where semimajor axis is more than X pc
        self.a_more_Xpc_idx = self._get_a_more_Xpc_idx()

        #timespan of analytical estimate
        self.analytical_tspan = self.orbit_params["t"][self.a_more_Xpc_idx]/myr - self.r_hard_time

        #ratio of G * density / inner velocity dispersion
        self.G_rho_per_sigma = get_G_rho_per_sigma(self.snaplist, self.r_hard_time, extent=self.r_infl)

        #semimajor axis hardening constant
        self.H = linear_fit_get_H(self.orbit_params["t"]/ketjugw.units.yr, self.orbit_params["a_R"]/ketjugw.units.pc, self.r_hard_time*1e6, self.analytical_tspan*1e6, self.G_rho_per_sigma)

       #eccentricity constant
        self.K = linear_fit_get_K(self.orbit_params["t"]/ketjugw.units.yr, self.orbit_params["e_t"], self.r_hard_time*1e6, self.analytical_tspan*1e6, self.H, self.G_rho_per_sigma, self.orbit_params["a_R"]/ketjugw.units.pc)

        #predicted a where GW emission dominates
        a_gr, t_gr = self.predict_gw_dominant_semimajoraxis(0.5)
        self.gw_dominant_semimajoraxis = {"a": a_gr, "t": t_gr}

        #predict the orbital parameters based off the eccentricity quantiles
        self.predicted_orbital_params = self._predicted_orbital_params_helper()

    #helper functions
    def _get_time_for_r(self, r, desc="r"):
        #find the time corresponding to a particular radial value [pc]
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
        #get the index of a value t within an array tarr
        #note tarr must have values in ascending order
        if not np.isnan(t):
            if t > tarr.max():
                warnings.warn("Value is larger than the largest array value. Returning index -1")
                return -1
            elif t < tarr.min():
                warnings.warn("Value is smaller than the smallest array value. Returning index 0")
                return 0
            else:
                return np.argmax(t < tarr)
        else:
            raise ValueError("t must not be nan")
    
    def _get_a_more_Xpc_idx(self):
        #determine where the semimajor axis decreases below gr_safe_radius
        idx = np.argmax(self.orbit_params["a_R"]/ketjugw.units.pc<self.gr_safe_radius)
        if idx < 2:
            idx = -1
        return idx
    
    def _predicted_orbital_params_helper(self):
        #compute the orbital parameters given different quantiles of ecc
        op = dict(
            t = {"lower":None, "median":None, "upper":None},
            a = {"lower":None, "median":None, "upper":None},
            e = {"lower":None, "median":None, "upper":None}
        )
        for k, q in zip(
                        ("lower", "median", "upper"), 
                        self.param_estimate_e_quantiles
                        ):
            op["t"][k], op["a"][k], op["e"][k] = self.compute_predict_orbital_params(q)
        return op
    
    def _get_ecc_for_idx(self, idx, avg=3):
        #determine the ecc at a given index, averaged over avg indices before
        #and after the desired index
        if idx is None:
            return np.nan
        start_idx = max(0, idx-avg)
        end_idx = min(len(self.orbit_params["e_t"])-1,idx+avg+1)
        return np.nanmedian(self.orbit_params["e_t"][start_idx:end_idx])
    
    #public functions
    def predict_gw_dominant_semimajoraxis(self, q):
        #predict where GW emission dominates, given the eccentricity quantile q,
        #starting from where a=a_h
        e0 = np.nanquantile(self.orbit_params["e_t"][self.r_hard_time_idx:self.a_more_Xpc_idx], q)
        a_gr, t_agr = gravitational_radiation_radius(
                                        self.bh_masses, self.r_hard, 
                                        self.r_hard_time, self.H,
                                        self.G_rho_per_sigma, e=e0)
        return a_gr, t_agr
    
    def compute_predict_orbital_params(self, q, idxs=None):
        """compute orbital params for a given initial eccentricity quantile"""
        if idxs is None:
            idxs = [self.r_hard_time_idx, self.a_more_Xpc_idx]
        assert isinstance(idxs, list)
        if idxs[1] != -1: assert idxs[0] < idxs[1]
        a0 = self.orbit_params["a_R"][idxs[0]]
        e0 = np.nanquantile(self.orbit_params["e_t"][idxs[0]:idxs[1]], q)
        t, a, e = analytic_evolve_peters_quinlan(a0, e0, self.orbit_params["t"][idxs[0]], self.orbit_params["t"][-1], self.orbit_params["m0"][0], self.orbit_params["m1"][0], self.G_rho_per_sigma, self.H, self.K)
        t /= myr
        a /= ketjugw.units.pc
        return t, a, e

    def plot(self, ax=None, add_radii=True, add_op_estimates=True, **kwargs):
        #plot the binary evolution, with points showing [r_infl, r_hard, 
        # analytical_tspan, gw radius] if add_radii==True, and estimates for 
        # the orbital parameters if add_op_estimates==True
        if ax is None:
            fig, ax = plt.subplots(2,1,sharex="all")
        binary_param_plot(self.orbit_params, ax=ax, toffset=self.time_offset, zorder=5)
        plt.suptitle(self.merger_name)
        if add_radii:
            ax[0].scatter(self.r_infl_time+self.time_offset, self.r_infl, zorder=10, label=r"$r_\mathrm{inf}$")
            ax[0].scatter(self.r_hard_time+self.time_offset, self.r_hard, zorder=10, label=r"$a_\mathrm{h}$")
            ax[0].axvspan(self.r_hard_time+self.time_offset, self.r_hard_time+self.analytical_tspan+self.time_offset, color="tab:red", alpha=0.4, label="H calculation")
            sc = ax[0].scatter(self.gw_dominant_semimajoraxis["t"]+self.time_offset, self.gw_dominant_semimajoraxis["a"], zorder=10, label=r"$a_\mathrm{GR}$")
            ax[0].axhline(self.gw_dominant_semimajoraxis["a"], ls=":", c=sc.get_facecolors())
            agr_low,_ = self.predict_gw_dominant_semimajoraxis(self.param_estimate_e_quantiles[0])
            agr_high,_ = self.predict_gw_dominant_semimajoraxis(self.param_estimate_e_quantiles[-1])
            ax[0].axhspan(agr_low, agr_high, facecolor=sc.get_facecolors(), alpha=0.4)
        if add_op_estimates:
            for i, (k,q) in enumerate(
                            zip(self.predicted_orbital_params["a"].keys(), 
                            self.param_estimate_e_quantiles)):
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
        #Print some of the key binary quantities
        print("BH Binary Quantities:")
        print("  Perturbation applied at {:.1f} Myr".format(self.time_offset))
        print("  H determined over a span of {:.1f} Myr".format(self.analytical_tspan))
        print("  G*rho/sigma: {:.3e}".format(self.G_rho_per_sigma))
        print("  Hardening rate H: {:.4f}".format(self.H))
        print("  Eccentricity rate K: {:.4f}".format(self.K))
        print("  Spin flip: {}".format(self.binary_spin_flip))
        if self.merged():
            print(self.merged)



class ChildSimData(BHBinaryData):
    def __init__(self) -> None:
        """
        A class that defines the fields which constitute the variables of 
        interest for the merger remannt. These properties are accessible to all 
        child classes, and also correspond to the fields which are loadable 
        from a hdf5 file. Those fields which are part of the inherited class
        are also part of this hdf5 file.
        """
        super().__init__()
        self.allowed_types = (int, float, str, bytes, np.int64, np.float64, np.ndarray, pygad.UnitArr, np.bool8, list, tuple)
        self.hdf5_file_name = None
    
    @property
    def parent_quantities(self):
        return self._parent_quantities
    
    @parent_quantities.setter
    def parent_quantities(self, v):
        assert isinstance(v, dict)
        self._parent_quantities = v

    @property
    def relaxed_stellar_velocity_dispersion(self):
        return self._relaxed_stellar_velocity_dispersion
    
    @relaxed_stellar_velocity_dispersion.setter
    def relaxed_stellar_velocity_dispersion(self, v):
        self._relaxed_stellar_velocity_dispersion = v
    
    @property
    def relaxed_stellar_velocity_dispersion_projected(self):
        return self._relaxed_stellar_velocity_dispersion_projected
    
    @relaxed_stellar_velocity_dispersion_projected.setter
    def relaxed_stellar_velocity_dispersion_projected(self, v):
        assert isinstance(v, dict)
        self._relaxed_stellar_velocity_dispersion_projected = v
    
    @property
    def relaxed_inner_DM_fraction(self):
        return self._relaxed_inner_DM_fraction
    
    @relaxed_inner_DM_fraction.setter
    def relaxed_inner_DM_fraction(self, v):
        self._relaxed_inner_DM_fraction = v
    
    @property
    def virial_info(self):
        return self._virial_info
    
    @virial_info.setter
    def virial_info(self, v):
        self._virial_info = v
    
    @property
    def relaxed_effective_radius(self):
        return self._relaxed_effective_radius
    
    @relaxed_effective_radius.setter
    def relaxed_effective_radius(self, v):
        assert isinstance(v, dict)
        self._relaxed_effective_radius = v
    
    @property
    def relaxed_half_mass_radius(self):
        return self._relaxed_half_mass_radius
    
    @relaxed_half_mass_radius.setter
    def relaxed_half_mass_radius(self, v):
        self._relaxed_half_mass_radius = v
    
    @property
    def relaxed_core_parameters(self):
        return self._relaxed_core_parameters
    
    @relaxed_core_parameters.setter
    def relaxed_core_parameters(self, v):
        self._relaxed_core_parameters = v

    @property
    def relaxed_density_profile(self):
        return self._relaxed_density_profile
    
    @relaxed_density_profile.setter
    def relaxed_density_profile(self, v):
        self._relaxed_density_profile = v

    @property
    def relaxed_density_profile_projected(self):
        return self._relaxed_density_profile_projected
    
    @relaxed_density_profile_projected.setter
    def relaxed_density_profile_projected(self, v):
        self._relaxed_density_profile_projected = v
    
    @property
    def relaxed_triaxiality_parameters(self):
        return self._relaxed_triaxiality_parameters
    
    @relaxed_triaxiality_parameters.setter
    def relaxed_triaxiality_parameters(self, v):
        self._relaxed_triaxiality_parameters = v
    
    @property
    def total_stellar_mass(self):
        return self._total_stellar_mass
    
    @total_stellar_mass.setter
    def total_stellar_mass(self, v):
        self._total_stellar_mass = v
    
    @property
    def ifu_map_ah(self):
        return self._ifu_map_ah
    
    @ifu_map_ah.setter
    def ifu_map_ah(self, v):
        assert isinstance(v, dict)
        self._ifu_map_ah = v
    
    @property
    def ifu_map_merger(self):
        return self._ifu_map_merger
    
    @ifu_map_merger.setter
    def ifu_map_merger(self, v):
        assert isinstance(v, dict)
        self._ifu_map_merger = v
    
    @property
    def snapshot_times(self):
        return self._snapshot_times
    
    @snapshot_times.setter
    def snapshot_times(self, v):
        self._snapshot_times = v
    
    @property
    def stellar_shell_inflow_velocity(self):
        return self._stellar_shell_inflow_velocity
    
    @stellar_shell_inflow_velocity.setter
    def stellar_shell_inflow_velocity(self, v):
        self._stellar_shell_inflow_velocity = v
    
    @property
    def bh_binary_watershed_velocity(self):
        return self._bh_binary_watershed_velocity
    
    @bh_binary_watershed_velocity.setter
    def bh_binary_watershed_velocity(self, v):
        self._bh_binary_watershed_velocity = v
    
    @property
    def beta_r(self):
        return self._beta_r
    
    @beta_r.setter
    def beta_r(self, v):
        assert isinstance(v, dict)
        self._beta_r = v

    @property
    def ang_mom_diff_angle(self):
        return self._ang_mom_diff_angle
    
    @ang_mom_diff_angle.setter
    def ang_mom_diff_angle(self, v):
        self._ang_mom_diff_angle = v
    
    @property
    def loss_cone(self):
        return self._loss_cone
    
    @loss_cone.setter
    def loss_cone(self, v):
        self._loss_cone = v

    @property
    def stars_in_loss_cone(self):
        return self._stars_in_loss_cone
    
    @stars_in_loss_cone.setter
    def stars_in_loss_cone(self, v):
        self._stars_in_loss_cone = v

    @property
    def particle_count(self):
        return self._particle_count
    
    @particle_count.setter
    def particle_count(self, v):
        assert isinstance(v, dict)
        self._particle_count = v
    
    @classmethod
    def load_from_file(cls, fname, decode="utf-8"):
        #first create a new class instance. At this stage, no properties are set
        C = cls()
        C.hdf5_file_name = fname

        #define some helpers
        def _recursive_dict_load(g):
            #recursively load a dictionary. Inspired from 3ML
            #g is a group object
            d = {}
            for key, val in g.items():
                if isinstance(val, h5py.Dataset):
                    tmp = val[()]
                    try:
                        d[key] = tmp.decode(decode)
                    except:
                        d[key] = tmp
                    #courtesy Elisa
                    if np.array_equal(d[key], "NONE_TYPE"):
                        d[key] = None
                elif isinstance(val, h5py.Group):
                    d[key] = _recursive_dict_load(val)
            return d
        
        def _main_setter(k, v):
            #set those class attributes which are datasets
            tmp = v[()]
            try:
                std_val = tmp.decode(decode)
            except:
                std_val = tmp
            if np.array_equal(std_val, "NONE_TYPE"):
                std_val = None
            if k == "logs":
                k = "_log"
            setattr(C, k, std_val)

        #now we need to recursively unpack the given hdf5 file
        with h5py.File(fname, mode="r") as f:
            for key, val in f.items():
                if isinstance(val, h5py.Dataset):
                    #these are top level datasets, and we don't expect there
                    #to be any
                    _main_setter(key, val)
                elif isinstance(val, h5py.Group):
                    #designed that datasets are grouped into two top-level 
                    #groups, so these need care unpacking
                    for kk, vv in val.items():
                        if isinstance(vv, h5py.Dataset):
                            _main_setter(kk, vv)
                        elif isinstance(vv, h5py.Group):
                            dict_val = _recursive_dict_load(vv)
                            setattr(C, kk, dict_val)
                        else:
                            ValueError("{}: Unkown type for unpacking!".format(kk))
        return C
    
    def _saver(self, g, l):
        #given a HDF5 group g, save all elements in list l
        #attributes defined with the @property method are not in __dict__, 
        #but their _members are. Append an underscore to all things in l
        l = ["_" + x for x in l]
        for attr in self.__dict__:
            if attr not in l:
                continue
            #now we strip the leading underscore if this should be saved
            attr = attr.lstrip("_")
            attr_val = getattr(self, attr)
            if isinstance(attr_val, self.allowed_types):
                g.create_dataset(attr, data=attr_val)
            elif attr is None:
                g.create_dataset(attr, "NONE_TYPE")
            elif isinstance(attr_val, dict):
                self._recursive_dict_save(g, attr_val, attr)
            else:
                raise ValueError("Error saving {}: cannot save {} type!".format(attr, type(attr_val)))

    def _recursive_dict_save(self, g, d, n):
        #recursively save a dictionary. Inspired from 3ML
        #g is group object, d is the dict, n is the new group name
        gnew = g.create_group(n)
        for key, val in d.items():
            if isinstance(val, self.allowed_types):
                gnew.create_dataset(key, data=val)
            elif val is None:
                gnew.create_dataset(key, data="NONE_TYPE")
            elif isinstance(val, dict):
                self._recursive_dict_save(gnew, val, key)
            else:
                raise ValueError("Error saving {}: cannot save {} type!".format(key, type(val)))
    
    #public functions
    def add_hdf5_field(self, n, val, field, fname=None):
        #add a new field to an existing HDF5 structure
        #n is attribute name, val is its value, field is where to save to, 
        #fname is the file name
        if fname is None:
            fname = self.hdf5_file_name
        field = field.rstrip("/")
        with h5py.File(fname, mode="a") as f:
            if isinstance(val, self.allowed_types):
                f.create_dataset(field+"/"+n, data=val)
            elif val is None:
                f.create_dataset(field+"/"+n, data="NONE_TYPE")
            elif isinstance(val, dict):
                # TODO this may not work...
                self._recursive_dict_save(f[field], val, n)
            else:
                raise ValueError("Error saving {}: cannot save {} type!".format(n, type(val)))
            self.add_to_log("Attribute {} has been added".format(n))
            f["/meta/logs"][...] = self._log
    
    def print_logs(self):
        print(self._log)



class ChildSim(BHBinary, ChildSimData):
    def __init__(self, paramfile, perturbID, gr_safe_radius=15, param_estimate_e_quantiles=[0.05, 0.5, 0.95], voronoi_kw={"Npx":300, "part_per_bin":5000}, shell_radius=3e-2, radial_edges=np.geomspace(2e-1,20,51), verbose=False) -> None:
        """
        A class which determines and sets the key merger remnant properties 
        from the raw simulation output data. 

        Parameters
        ----------
        paramfile: see BHBinary 
        perturbID: see BHBinary
        gr_safe_radius: see BHBinary
        param_estimate_e_quantiles: see BHBinary
        voronoi_kw: dict of values used for Voronoi tesselation, which is 
                    passed to voronoi_binned_los_V_statistics()
        shell_radius: radius [in kpc] of a shell through which crossing
                      statistics are computed
        radial_edges: sequence of values specifying the edge of the radial bins
                      used for density profiles, beta profile [in kpc]
        verbose: bool, verbose printing?
        """
        self.verbose = verbose
        if self.verbose:
            print("> Determining binary quantities")
        super().__init__(paramfile, perturbID, gr_safe_radius, param_estimate_e_quantiles)
        self.voronoi_kw = voronoi_kw
        self.shell_radius = shell_radius
        self.radial_edges = radial_edges
        self.merged_idx = -1

        #set the properties
        #set some key propertie from the parent run
        pfv = read_parameters(paramfile, verbose=False)
        self.parent_quantities = dict(
            perturb_time = pfv.perturbTime,
            initial_e = pfv.e,
            r0 = pfv.r0,
            rperi = pfv.rperi,
            time_to_peri = pfv.time_to_pericenter,
            rvir = pfv.virial_radius
        )

        #set the particle counts
        if self.verbose:
            print("> Determining particle counts")
        self.particle_count = dict(
                    stars = len(self.main_snap.stars["mass"]),
                    dm = len(self.main_snap.dm["mass"]),
                    bh = len(self.main_snap.bh["mass"])
        )
        
        #get the stellar velocity dispersions
        if self.verbose:
            print("> Determining stellar velocity dispersion")
        self.relaxed_stellar_velocity_dispersion = np.nanstd(self.main_snap.stars["vel"], axis=0)

        #projected effective radius, velocity dispersion in 1Re, and density 
        if self.verbose:
            print("> Determining projected quantities")
        # TODO check Re! Is very large
        self.relaxed_effective_radius, self.relaxed_stellar_velocity_dispersion_projected, self.relaxed_density_profile_projected = self._get_projected_quantities()

        #DM fraction within 1Re
        if self.verbose:
            print("> Determining inner DM fraction")
        self.relaxed_inner_DM_fraction = inner_DM_fraction(self.main_snap, Re=self.relaxed_effective_radius["estimate"])

        #3D half mass radius
        if self.verbose:
            print("> Determining half mass radius")
        self.relaxed_half_mass_radius = pygad.analysis.profile_dens(self.main_snap.stars, "mass", center=self.main_snap_mass_centre)

        #total stellar mass of the remnant
        if self.verbose:
            print("> Determining total stellar mass")
        self.total_stellar_mass = self.main_snap.stars["mass"][0] * len(self.main_snap.stars)

        #3D density profile of relaxed remnant
        if self.verbose:
            print("> Determining 3D density profile")
        self.relaxed_density_profile = pygad.analysis.profile_dens(self.main_snap.stars, "mass", r_edges=self.radial_edges, center=self.main_snap_mass_centre)

        #relaxed core parameters
        if self.verbose:
            print("> Determining core fit parameters")
        try:
            r_fit_range = get_histogram_bin_centres(self.radial_edges)
            self.relaxed_core_parameters = fit_Terzic05_profile(r_fit_range, self.relaxed_density_profile, self.relaxed_effective_radius["estimate"], max_nfev=1000)
        except RuntimeError:
            self.add_to_log("Core-fit parameters coudl not be determined! Skipping...")
            self.relaxed_core_parameters = {"rhob": np.nan, "rb": np.nan, "n": np.nan, "g": np.nan, "b": np.nan, "a": np.nan}

        #triaxiality parameters of relaxed remnant
        if self.verbose:
            print("> Determining triaxiality parameters at merger")
        self.relaxed_triaxiality_parameters = self._triaxial_helper()

        #IFU data at time when a = a_h
        if self.verbose:
            print("> Determining IFU data at time of a=a_h")
        self.ifu_map_ah = self.create_ifu_map(t=self.r_hard_time)

        #IFU data at merger
        if self.verbose:
            print("> Determining IFU data at time of merger / last snap")
        self.ifu_map_merger = self.create_ifu_map(idx=self.merged_idx)

        #list of all snapshot times, waterhsed velocity, stellar inflow 
        # velocity, angle between galaxy stellar ang mom and BH ang mom,
        #num stars in loss cone
        if self.verbose:
            print("> Determining time series data")
        self.snapshot_times, self.bh_binary_watershed_velocity, self.stellar_shell_inflow_velocity, self.ang_mom_diff_angle, self.loss_cone, self.stars_in_loss_cone = self._get_time_series_data(R=self.shell_radius)

        #velocity anisotropy of remnant as a function of radius
        if self.verbose:
            print("> Determining beta profile")
        self.beta_r = self._beta_r_helper()
        
        #virial information of relaxed remnant
        if self.verbose:
            print("> Determining virial information")
        self.virial_info = {"radius":None, "mass":None}
        self.virial_info["radius"], self.virial_info["mass"] = pygad.analysis.virial_info(self.main_snap, center=self.main_snap_mass_centre)
    
    #define the "main" snapshot we will be working with, i.e. the one
    #closest to merger, and move it to CoM coordinates
    @cached_property
    def main_snap(self):
        if self.merged():
            self.merged_idx = snap_num_for_time(self.snaplist, self.merged.time, method="nearest")
        else:
            self.merged_idx = -1
        snap = pygad.Snapshot(self.snaplist[self.merged_idx], physical=True)
        xcom = get_com_of_each_galaxy(snap, verbose=False)
        vcom = get_com_velocity_of_each_galaxy(snap, xcom, verbose=False)
        snap["pos"] -= list(xcom.values())[0]
        snap["vel"] -= list(vcom.values())[0]
        return snap
    
    @cached_property
    def main_snap_mass_centre(self):
        centre_guess = pygad.analysis.center_of_mass(self.main_snap.bh)
        return pygad.analysis.shrinking_sphere(self.main_snap.stars, centre_guess, 20)
    
    #helper functions
    def _get_projected_quantities(self):
        Re, vsig, rho = projected_quantities(self.main_snap, r_edges=self.radial_edges)
        return list(Re.values())[0], list(vsig.values())[0], list(rho.values())[0]
    
    def _triaxial_helper(self):
        ratios = dict(
            ba = np.full(len(self.radial_edges)-1, np.nan),
            ca = np.full(len(self.radial_edges)-1, np.nan)
        )
        radii_to_mask = list(zip(self.radial_edges[:-1], self.radial_edges[1:]))
        for i, r in enumerate(radii_to_mask):
            radial_mask = get_radial_mask(self.main_snap, r, self.main_snap_mass_centre)
            ratios["ba"][i], ratios["ca"][i] = get_galaxy_axis_ratios(self.main_snap, self.main_snap_mass_centre, radial_mask=radial_mask)
        return ratios
    
    def _get_time_series_data(self, R=3e-2):
        #cycle through all snaps extracting key values to create a time series
        bound_snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][0]/myr)
        last_snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][-1]/myr)
        t = np.full(last_snap_idx-bound_snap_idx+1, np.nan)
        w = np.full_like(t, np.nan)
        v = {}
        theta = np.full_like(t, np.nan)
        J_lc = np.full_like(t, np.nan)
        num_J_lc = np.full_like(t, np.nan)
        m_min = min([self.orbit_params["m0"][0], self.orbit_params["m1"][0]])
        for i, j in enumerate(np.arange(bound_snap_idx, last_snap_idx+1)):
            snap = pygad.Snapshot(self.snaplist[j], physical=True)
            #as we are interested in flow rates about binary, set binary as
            #the centre of mass
            this_centre = pygad.analysis.center_of_mass(snap.bh)
            t[i] = convert_gadget_time(snap, new_unit="Myr")
            idx = self._get_idx_in_vec(t[i], self.orbit_params["t"]/myr)
            w[i] = 0.85 * np.sqrt(m_min / self.orbit_params["a_R"][idx]) / ketjugw.units.km_per_s
            v["{:.1f}".format(t[i])] = shell_flow_velocities(snap.stars, R, centre=this_centre, direction="in")
            #determine angle difference in J
            theta[i] = angular_momentum_difference_gal_BH(snap)
            #determine loss cone J, _a is semimajor axis from ketjugw as pygad
            #scalar object
            _a = pygad.UnitScalar(self.orbit_params["a_R"][idx], "pc")
            J_lc[i] = loss_cone_angular_momentum(snap, _a)
            num_J_lc[i] = np.sum(snap.stars["angmom"] < J_lc[i])
            snap.delete_blocks()
        return t, w, v, theta, J_lc, num_J_lc
    
    def _beta_r_helper(self):
        vspherical = spherical_components(self.main_snap.stars["pos"], self.main_snap.stars["vel"])
        beta_r, radbins, bincount = beta_profile(self.main_snap.stars["r"], vspherical, 0.05)
        return {"beta_r": beta_r, "radbins":radbins, "bincount": bincount}

    #public functions
    def plot(self):
        warnings.warn("This method is not available for this class. If you wish to plot the BH Binary orbital parameters, use the binary_plot() method.")
    
    def binary_plot(self, ax=None, add_radii=True, add_op_estimates=True, **kwargs):
        super().plot(ax, add_radii, add_op_estimates, **kwargs)
    
    def create_ifu_map(self, t=None, r=None, idx=None):
        if idx is None and t is not None:
            idx = snap_num_for_time(self.snaplist, t)
        elif idx is None and t is None:
            raise RuntimeError("One of t (time) or idx (snapshot number) must be specified!")

        snap = pygad.Snapshot(self.snaplist[idx], physical=True)
        mass_centre = pygad.analysis.center_of_mass(snap.bh)
        # TODO is the best radius to use?
        if r is None:
            r = self.relaxed_effective_radius["estimate"]
        ball_mask = pygad.BallMask(r, center=mass_centre)
        subsnap = snap.stars[ball_mask]
        return voronoi_binned_los_V_statistics(
                    subsnap["pos"][:,0], subsnap["pos"][:,1], 
                    subsnap["vel"][:,2], subsnap["mass"], **self.voronoi_kw)
    
    def print(self):
        #print the currently loaded attributes of the data cube
        s = "Data Cube has the following properties loaded:\n"
        keys = self.__dict__.keys()
        for k in keys:
            if k[0] != "_":
                s += " > {}\n".format(k)
        print(s)
    
    def make_hdf5(self, fname, exist_ok=False):
        if os.path.isfile(fname) and not exist_ok:
            raise ValueError("HDF5 file already exists!")
        with h5py.File(fname, mode="w") as f:
            #set up some meta data
            meta = f.create_group("meta")
            now = datetime.datetime.now()
            meta.attrs["created"] = now.strftime(date_str)
            meta.attrs["created_by"] = username
            meta.attrs["last_accessed"] = now.strftime(date_str)
            meta.attrs["last_user"] = username
            meta.create_dataset("logs", data=self._log)
        
            #save the binary info
            bhb = f.create_group("bh_binary")
            data_list = ["bh_masses", "bh_formation_time", "binary_merger_timescale", "r_infl", "r_infl_time", "r_infl_ecc", "r_bound", "r_bound_time", "r_bound_ecc", "r_hard", "r_hard_time", "r_hard_ecc", "analytical_tspan", "G_rho_per_sigma", "H", "K", "gw_dominant_semimajoraxis", "predicted_orbital_params", "formation_ecc_spread", "binary_merger_remnant", "binary_spin_flip"]
            self._saver(bhb, data_list)
            f["/bh_binary/predicted_orbital_params"].attrs["quantiles"] = self.param_estimate_e_quantiles

            #save the galaxy properties
            gp = f.create_group("galaxy_properties")
            data_list = ["relaxed_stellar_velocity_dispersion", "relaxed_stellar_velocity_dispersion_projected", "relaxed_inner_DM_fraction", "virial_info", "relaxed_effective_radius", "relaxed_half_mass_radius", "relaxed_core_parameters", "relaxed_triaxiality_parameters", "relaxed_density_profile", "total_stellar_mass", "ifu_map_ah", "ifu_map_merger", "snapshot_times", "stellar_shell_inflow_velocity", "bh_binary_watershed_velocity", "beta_r", "ang_mom_diff_angle", "loss_cone", "stars_in_loss_cone", "particle_count"]
            self._saver(gp, data_list)
            f["/galaxy_properties/stellar_shell_inflow_velocity"].attrs["shell_radius"] = self.shell_radius
            f["/galaxy_properties/relaxed_density_profile"].attrs["radial_edges"] = self.radial_edges

            #save the parent properties
            gP = f.create_group("parent_properties")
            self._saver(gP, ["parent_quantities"])

