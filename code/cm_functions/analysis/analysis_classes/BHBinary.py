import os.path
import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pygad
import ketjugw

from .BHBinaryData import BHBinaryData
from ..analyse_snap import *
from ..general import snap_num_for_time
from ..orbit import get_bound_binary, linear_fit_get_H, linear_fit_get_K, analytic_evolve_peters_quinlan
from ...general import xval_of_quantity, convert_gadget_time
from ...mathematics import iqr
from ...plotting import binary_param_plot
from ...utils import read_parameters, get_ketjubhs_in_dir, get_snapshots_in_dir
from ...env_config import date_format, username, _cmlogger

__all__ = ["myr", "BHBinary"]

_logger = _cmlogger.copy(__file__)

myr = ketjugw.units.yr * 1e6
hubble_time = 13800

class BHBinary(BHBinaryData):
    def __init__(self, paramfile, perturbID, apfile) -> None:
        """
        A class which determines and sets the key BH binary properties from the
        raw simulation output data.

        Parameters
        ----------
        paramfile : str
            path to parameter file corresponding to the merger run
        perturbID : str
            name of the perturbation directory, e.g. 000
        apfile : str
            path to parameter file for analysis

        Raises
        ------
        ValueError
            when a radius (rinfl, rbound, rhard) cannot be determined, or when
            the data path does not exist
        """
        super().__init__()
        self.merger_pars = read_parameters(paramfile)
        self.analysis_pars = read_parameters(apfile)
        if self.merger_pars["file_locations"]["perturb_sub_dir"] is None:
            # we are dealing with a non-perturbed simulation set
            data_path = os.path.join(self.merger_pars["file_locations"]["save_location"], perturbID, "output")
            self.merger_name = f"{self.merger_pars['calculated']['full_save_location'].rstrip('/').split('/')[-2]}-{perturbID}"
        else:
            # we are dealing with a perturbed simulation set
            data_path = os.path.join(self.merger_pars["calculated"]["full_save_location"], self.merger_pars["file_locations"]["perturb_sub_dir"], perturbID, "output")
            self.merger_name = f"{self.merger_pars['calculated']['full_save_location'].rstrip('/').split('/')[-1]}-{perturbID}"
        if not os.path.isdir(data_path):
            raise ValueError("The data path does not exist!")
        try:
            self.bhfile = get_ketjubhs_in_dir(data_path)[0]
        except IndexError:
            _logger.logger.exception("No ketju_bhs.hdf5 file!")
            raise
        self.snaplist = get_snapshots_in_dir(data_path)
        self.gr_safe_radius = self.analysis_pars["bh_binary"]["target_semimajor_axis"]["value"]
        bh1, bh2, self.merged = get_bound_binary(self.bhfile)
        self.orbit_params = ketjugw.orbit.orbital_parameters(bh1, bh2)
        self.time_offset = self.merger_pars["perturb_properties"]["perturb_time"]["value"]
        if self.merger_pars["perturb_properties"]["perturb_time"]["unit"] == "Gyr":
            self.time_offset *= 1e3
        self.param_estimate_e_quantiles = self.analysis_pars["bh_binary"]["eccentricity_quantiles_for_estimates"]

        #set the properties
        snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][0]/myr+convert_gadget_time(pygad.Snapshot(self.snaplist[0], physical=True), "Myr"), method="ceil")
        snap = pygad.Snapshot(self.snaplist[snap_idx], physical=True)
        #black hole mass
        self.bh_masses = snap.bh["mass"]

        # stellar mass
        self.stellar_mass = snap.stars["mass"][0]

        #time when binary is 'formed'
        self.binary_formation_time = self.time_offset + self.orbit_params["t"][0] / myr

        #merger timescale
        if self.merged:
            self.binary_lifetime_timescale = (self.orbit_params["t"][-1] - self.orbit_params["t"][0]) / myr
        else:
            self.binary_lifetime_timescale = np.nan
        
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
        #self.r_infl = list(influence_radius(snap).values())[0].in_units_of("pc") #in pc
        _r_infl = []
        for sfile in self.snaplist[snap_idx:]:
            s = pygad.Snapshot(sfile, physical=True)
            _r_infl.append(
                list(influence_radius(snap).values())[0].in_units_of("pc") #in pc
            )
            s.delete_blocks()
            del s
        self.r_infl = pygad.UnitScalar(np.nanmedian(_r_infl), units=_r_infl[0].units)


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
        except ValueError:
            _logger.logger.warning("r_hard_time_idx not valid -> setting to 0")
            self.r_hard_time_idx = 0

        #index where semimajor axis is more than X pc
        self.a_more_Xpc_idx = self._get_a_more_Xpc_idx()

        #timespan of analytical estimate
        self.analytical_tspan = pygad.UnitScalar(self.orbit_params["t"][self.a_more_Xpc_idx]/myr - self.r_hard_time, units="Myr")

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
            tval = xval_of_quantity(r, self.orbit_params["t"]/myr, self.orbit_params["a_R"]/ketjugw.units.pc, xsorted=True)
            return pygad.UnitScalar(tval, units="Myr")
        except ValueError:
            #when the radius is reached is not covered by the data
            if r > self.orbit_params["a_R"][0] / ketjugw.units.pc:
                _logger.logger.warning(f"Time of {desc} cannot be estimated from the data, as it occurs before the binary is bound! This point will be omitted from further analysis and plots.")
            elif r < self.orbit_params["a_R"][-1] / ketjugw.units.pc:
                _logger.logger.warnings(f"Time of {desc} cannot be estimated from the data, as it occurs after the last available data point! This point will be omitted from further analysis and plots.")
            else:
                raise ValueError(f"{desc} cannot be determined")
            return np.nan

    def _get_idx_in_vec(self, t, tarr):
        #get the index of a value t within an array tarr
        #note tarr must have values in ascending order
        if not np.isnan(t):
            if t > tarr[-1]:
                raise ValueError("Value is larger than the largest array value.")
            elif t < tarr[0]:
                raise ValueError("Value is smaller than the smallest array value")
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
            ttemp, atemp, op["e"][k] = self.compute_predict_orbital_params(q)
            op["t"][k], op["a"][k] = pygad.UnitArr(ttemp, units="Myr"), pygad.UnitArr(atemp, units="pc")
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
        return pygad.UnitScalar(a_gr, units="pc"), pygad.UnitScalar(t_agr, units="Myr")
    
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
                    l = ax[0].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["a"][k], ls=":", label=f"{q:.2f} quantile", **kwargs)
                else:
                    ax[0].plot(self.predicted_orbital_params["t"][k]+self.time_offset, self.predicted_orbital_params["a"][k], ls=("--" if i==1 else ":"), c=l[-1].get_color(), label=f"{q:.2f} quantile", **kwargs)
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
        print(f"  Perturbation applied at {self.time_offset:.1f} Myr")
        print(f"  H determined over a span of {self.analytical_tspan:.1f} Myr")
        print(f"  G*rho/sigma: {self.G_rho_per_sigma:.3e}")
        print(f"  Hardening rate H: {self.H:.4f}")
        print(f"  Eccentricity rate K: {self.K:.4f}")
        print(f"  Spin flip: {self.binary_spin_flip}")
        if self.merged():
            print(self.merged)
    
    def _make_hdf5_helper(self, f):
        # helper method for saving hdf5 files that works with inheritance
        # f is a file stream
        bhb = f.create_group("bh_binary")
        data_list = [
            "bh_masses",
            "stellar_mass",
            "binary_formation_time",
            "binary_lifetime_timescale",
            "r_infl",
            "r_infl_time",
            "r_infl_ecc",
            "r_bound",
            "r_bound_time",
            "r_bound_ecc",
            "r_hard",
            "r_hard_time",
            "r_hard_ecc",
            "analytical_tspan",
            "G_rho_per_sigma",
            "H",
            "K",
            "gw_dominant_semimajoraxis",
            "predicted_orbital_params",
            "formation_ecc_spread",
            "binary_merger_remnant",
            "binary_spin_flip"]
        self._saver(bhb, data_list)
        self._add_attr(f["/bh_binary/predicted_orbital_params"], "quantiles", self.param_estimate_e_quantiles)

        #set up some meta data
        meta = f.create_group("meta")
        now = datetime.datetime.now()
        self._add_attr(meta, "merger_name", self.merger_name)
        self._add_attr(meta, "created", now.strftime(date_format))
        self._add_attr(meta, "created_by", username)
        self._add_attr(meta, "last_accessed", now.strftime(date_format))
        self._add_attr(meta, "last_user", username)
        meta.create_dataset("logs", data=self._log)

    def make_hdf5(self, fname, exist_ok=False):
        if os.path.isfile(fname) and not exist_ok:
            raise ValueError("HDF5 file already exists!")
        with h5py.File(fname, mode="w") as f:
            #save the binary info
            self._make_hdf5_helper(f)
