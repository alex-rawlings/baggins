import datetime
from functools import cached_property
import os.path
import warnings
import h5py
import numpy as np
import pygad
import ketjugw

from . import BHBinary, ChildSimData, myr, date_str
from ..analyse_snap import *
from ..general import beta_profile, snap_num_for_time
from ..masks import get_binding_energy_mask
from ..voronoi import voronoi_binned_los_V_statistics
from ...general import convert_gadget_time
from ...literature import fit_Terzic05_profile
from ...mathematics import get_histogram_bin_centres, spherical_components
from ...utils import read_parameters
from ...env_config import username

__all__ = ["ChildSim"]

class ChildSim(BHBinary, ChildSimData):
    def __init__(self, paramfile, perturbID, apfile, verbose=False) -> None:
        """
        A class which determines and sets the key merger remnant properties 
        from the raw simulation output data.

        Parameters
        ----------
        paramfile : str
            see BHBinary 
        perturbID : str
            see BHBinary 
        apfile : str
            see BHBinary 
        verbose : bool, optional
            verbose printing?, by default False
        """
        self.verbose = verbose
        if self.verbose:
            print("> Determining binary quantities")
        super().__init__(paramfile, perturbID, apfile)
        afv = read_parameters(apfile)
        #set the analysis parameters from the separate file
        self.galaxy_radius = afv.galaxy_radius
        self.com_consistency = afv.com_consistency
        self.relaxed_criteria = afv.relaxed_criteria
        self.voronoi_kw = afv.voronoi_kw
        self.shell_radius = pygad.UnitScalar(float(afv.shell_radius), units="pc")
        self.radial_edges = {}
        self.radial_bin_centres = {}
        for k, v in afv.radial_edges.items():
            self.radial_edges[k] = pygad.UnitArr(v, units="pc")
            self.radial_bin_centres[k] = pygad.UnitArr(get_histogram_bin_centres(v), units=self.radial_edges[k].units)
        self.merged_or_last_idx = -1
        self.galaxy_radius_mask = self.galaxy_radius
        self._energy_units = None

        #set the main properties
        #set some key properties from the parent run
        pfv = read_parameters(paramfile, verbose=False)
        self.parent_quantities = dict(
            perturb_time = pfv.perturbTime * 1e3,
            initial_e = pfv.e,
            r0 = pfv.r0 * 1e3,
            rperi = pfv.rperi * 1e3,
            time_to_peri = pfv.time_to_pericenter * 1e3,
            rvir = pfv.virial_radius * 1e3
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
        self.relaxed_stellar_velocity_dispersion = np.nanstd(self.main_snap.stars[self.galaxy_radius_mask]["vel"], axis=0)

        #projected effective radius, velocity dispersion in 1Re, and density 
        if self.verbose:
            print("> Determining projected quantities")
        self.relaxed_effective_radius, self.relaxed_stellar_velocity_dispersion_projected, self.relaxed_density_profile_projected = self._get_projected_quantities()

        #DM fraction within 1Re
        if self.verbose:
            print("> Determining inner DM fraction")
        self.relaxed_inner_DM_fraction = inner_DM_fraction(self.main_snap, Re=self.relaxed_effective_radius["estimate"])

        #3D half mass radius
        if self.verbose:
            print("> Determining half mass radius")
        self.relaxed_half_mass_radius = pygad.analysis.half_mass_radius(self.main_snap.stars[self.galaxy_radius_mask])

        #total stellar mass of the remnant
        if self.verbose:
            print("> Determining total stellar mass")
        self.total_stellar_mass = pygad.UnitScalar(self.main_snap.stars["mass"][0] * len(self.main_snap.stars[self.galaxy_radius_mask]), units=self.main_snap["mass"].units)

        #3D density profile of relaxed remnant
        if self.verbose:
            print("> Determining 3D density profile")
        self.relaxed_density_profile = self._dens_prof_helper()

        #relaxed core parameters
        if self.verbose:
            print("> Determining core fit parameters")
        try:
            self.relaxed_core_parameters = fit_Terzic05_profile(self.radial_bin_centres["stars"], self.relaxed_density_profile["stars"], self.relaxed_effective_radius["estimate"], max_nfev=1000)
        except RuntimeError:
            self.add_to_log("Core-fit parameters could not be determined! Skipping...")
            self.relaxed_core_parameters = {"rhob": np.nan, "rb": np.nan, "n": np.nan, "g": np.nan, "b": np.nan, "a": np.nan}

        #triaxiality parameters of relaxed remnant
        if self.verbose:
            print("> Determining triaxiality parameters at merger")
        self.relaxed_triaxiality_parameters, self.binding_energy_bins = self._triaxial_helper()

        #IFU data at time when a = a_h
        if self.verbose:
            print("> Determining IFU data at time of a=a_h")
        self.ifu_map_ah = self.create_ifu_map(t=self.r_hard_time)

        #IFU data at merger
        if self.verbose:
            print("> Determining IFU data at time of merger / last snap")
        self.ifu_map_merger = self.create_ifu_map(idx=self.merged_or_last_idx)

        # list of all snapshot times, waterhsed velocity, stellar inflow 
        # velocity, angle between galaxy stellar ang mom and BH ang mom,
        # num stars in loss cone, ang mom vectors, num escaping stars
        if self.verbose:
            print("> Determining time series data")
        self.snapshot_times, self.bh_binary_watershed_velocity, self.stellar_shell_inflow_velocity, self.ang_mom_diff_angle, self.loss_cone, self.stars_in_loss_cone, self.ang_mom, self.num_escaping_stars = self._get_time_series_data(R=self.shell_radius)

        #velocity anisotropy of remnant as a function of radius
        if self.verbose:
            print("> Determining beta profile")
        self.beta_r = self._beta_r_helper()
    

    #set auxillary properties that we just need for this class, but don't
    #want to save to the hdf5 file
    #define the "main" snapshot we will be working with, i.e. the one
    #closest to merger, and move it to CoM coordinates
    @cached_property
    def main_snap(self):
        if self.merged():
            self.merged_or_last_idx = snap_num_for_time(self.snaplist, self.merged.time, method="floor")
        else:
            self.merged_or_last_idx = -1
        if self.verbose:
            print(f"Merger remnant snapshot is number {self.merged_or_last_idx}")
        snap = pygad.Snapshot(self.snaplist[self.merged_or_last_idx], physical=True)
        xcom = get_com_of_each_galaxy(snap, method="ss", family="stars", verbose=False)
        vcom = get_com_velocity_of_each_galaxy(snap, xcom, verbose=False)
        # do a number of tests to determine if the remnant is relaxed

        # TEST 1
        # ensure the difference in com position and velocity using different BH
        # values as an initial guess are consistent -> may not be relaxed 
        # otherwise
        xcom_None_bool = [v is None for v in xcom.values()]
        vcom_None_bool = [v is None for v in vcom.values()]
        self.relaxed_remnant_flag = True
        if not np.any(xcom_None_bool) and not np.any(vcom_None_bool):
            # No Nones in CoM dictionaries. Check consistency
            msg = "Total CoM estimate for {} differs when using the individual BHs as an initial guess. The merger remnant may not be relaxed, leading to incorrect estimates that rely on centring."
            if np.any(np.abs(np.diff(list(xcom.values()), axis=0)) > self.com_consistency["pos"] / 1e3):
                warnings.warn(msg.format("position"))
                self.add_to_log(msg.format("position"))
                self.relaxed_remnant_flag = False
            if np.any(np.abs(np.diff(list(vcom.values()), axis=0)) > self.com_consistency["vel"]):
                warnings.warn(msg.format("velocity"))
                self.add_to_log(msg.format("velocity"))
                self.relaxed_remnant_flag = False
        # move to the expected CoM frame
        massive_BH_id = get_massive_bh_ID(snap.bh)
        # use potential minimum as CoM
        xcom = get_com_of_each_galaxy(snap, method="pot", verbose=False)
        vcom = get_com_velocity_of_each_galaxy(snap, xcom=xcom, verbose=False)
        snap["pos"] -= xcom[massive_BH_id]
        snap["vel"] -= vcom[massive_BH_id]

        # TEST 2
        # these are from https://ui.adsabs.harvard.edu/abs/2007MNRAS.381.1450N/abstract (Neto+07)
        # and rely on virial properties, so lets just set those values here too
        if self.verbose:
            print("> Determining virial information")
        self.virial_info = {"radius":None, "mass":None}
        self.virial_info["radius"], self.virial_info["mass"] = pygad.analysis.virial_info(snap)
        virial_mask = pygad.BallMask(self.virial_info["radius"])
        s = np.linalg.norm(pygad.analysis.center_of_mass(snap[virial_mask])) / self.virial_info["radius"]
        if s > self.relaxed_criteria["sep"]:
            self.relaxed_remnant_flag = False
        if virial_ratio(snap[virial_mask]) > self.relaxed_criteria["vrat"]:
            self.relaxed_remnant_flag = False
        # return the snap irrespectively
        return snap

    @property
    def galaxy_radius_mask(self):
        return self._galaxy_radius_mask
    
    @galaxy_radius_mask.setter
    def galaxy_radius_mask(self, v):
        v = pygad.UnitScalar(v, units="pc")
        self._galaxy_radius_mask = pygad.BallMask(v)
    
    #helper functions
    def _dens_prof_helper(self):
        rhos = dict.fromkeys(self.radial_bin_centres, np.nan)
        # 3D density profile for stars
        rhos["stars"] = pygad.analysis.profile_dens(self.main_snap.stars[self.galaxy_radius_mask], "mass", r_edges=self.radial_edges["stars"])
        # 3D density profile for DM
        rhos["dm"] = pygad.analysis.profile_dens(self.main_snap.dm, "mass", r_edges=self.radial_edges["dm"])
        # 3D density profile for ALL
        rhos["all"] = pygad.analysis.profile_dens(self.main_snap, "mass", r_edges=self.radial_edges["all"])
        return rhos


    def _get_projected_quantities(self):
        Re, vsig, rho = projected_quantities(self.main_snap[self.galaxy_radius_mask], r_edges=self.radial_edges["stars"])
        return list(Re.values())[0], list(vsig.values())[0], list(rho.values())[0]
    
    def _triaxial_helper(self):
        ratios = dict(
            ba = np.full(20, np.nan),
            ca = np.full(20, np.nan)
        )
        try:
            energy_mask_gen = get_binding_energy_mask(self.main_snap[self.galaxy_radius_mask], family="stars")
            i = 0
            while True:
                try:
                    this_energy_mask, energy_bins, self._energy_units = next(energy_mask_gen)
                except StopIteration:
                    break
                ratios["ba"][i], ratios["ca"][i] = get_galaxy_axis_ratios(self.main_snap[self.galaxy_radius_mask], bin_mask=this_energy_mask)
                i += 1
        except AssertionError:
            energy_bins = np.nan
            for i in range(len(ratios["ba"])):
                ratios["ba"][i] = np.nan
                ratios["ca"][i] = np.nan
            self._energy_units = None
        return ratios, energy_bins
    
    def _get_time_series_data(self, R=30, tspan=0.1):
        R = pygad.UnitScalar(R, units="pc")
        # average ketjugw quantities over a small interval
        delta_idx = int(np.floor(0.5 * tspan/(self.orbit_params["t"][1] - self.orbit_params["t"][0])/myr))
        #cycle through all snaps extracting key values to create a time series
        bound_snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][0]/myr)
        last_snap_idx = snap_num_for_time(self.snaplist, self.orbit_params["t"][-1]/myr)
        t = np.full(last_snap_idx-bound_snap_idx+1, np.nan)
        w = np.full_like(t, np.nan)
        v = {}
        theta = np.full_like(t, np.nan)
        J_lc = np.full_like(t, np.nan)
        num_J_lc = np.full_like(t, np.nan)
        bh_stars_J = {"bh":np.full((len(t),3), np.nan), "stars":np.full((len(t),3), np.nan)}
        num_vescs = np.full_like(t, np.nan)
        prev_vescs = []
        m_min = min([self.orbit_params["m0"][0], self.orbit_params["m1"][0]])
        J_unit = self.main_snap["angmom"].units
        for i, j in enumerate(np.arange(bound_snap_idx, last_snap_idx+1)):
            snap = pygad.Snapshot(self.snaplist[j], physical=True)
            # as we are interested in flow rates about binary, set binary as
            # the centre of mass
            snap["pos"] -= pygad.analysis.center_of_mass(snap.bh)
            snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.bh, "vel")
            t[i] = convert_gadget_time(snap, new_unit="Myr")
            try:
                idx = self._get_idx_in_vec(t[i], self.orbit_params["t"]/myr)
                _a = np.nanmedian(self.orbit_params["a_R"][idx-delta_idx:idx+delta_idx])
                _e = np.nanmedian(self.orbit_params["e_t"][idx-delta_idx:idx+delta_idx])
                # set the watershed velocity
                w[i] = 0.85 * np.sqrt(m_min / _a) / ketjugw.units.km_per_s
                # determine loss cone J, _a_pyagd is semimajor axis from 
                # ketjugw as pygad scalar object
                _a_pygad = pygad.UnitScalar(_a/ketjugw.units.pc, "pc")
                J_lc[i] = loss_cone_angular_momentum(snap, _a_pygad, e=_e)
                # determine the magnitude of the angular momentum
                star_J_mag = pygad.utils.geo.dist(snap.stars[self.galaxy_radius_mask]["angmom"])
                num_J_lc[i] = np.sum(star_J_mag < J_lc[i])
            except ValueError:
                w[i] = np.nan
                J_lc[i] = np.nan
                num_J_lc[i] = np.nan
            # determine the inflow velocities of stars through the shell
            v[f"{t[i]:07.1f}"] = shell_flow_velocities(snap.stars, R, direction="in")
            #determine angle difference in J
            theta[i], bh_stars_J["bh"][i,:], bh_stars_J["stars"][i,:] = angular_momentum_difference_gal_BH(snap, mask=self.galaxy_radius_mask)
            # determine the number of hypervelocity stars
            num_vescs[i], prev_vescs = count_new_hypervelocity_particles(snap, prev=prev_vescs)
            # save memory
            snap.delete_blocks()
        # ensure smoothing og ketjugw quantities is over an interval smaller 
        # than the snapshot interval
        if t[1]-t[0] > tspan:
            warnings.warn(f"ketjugw quantities averaged over an interval {tspan:.2f} Myr that is larger than the snapshot interval {t[1]-t[0]:.2f}!")
        return pygad.UnitArr(t, units="Myr"), pygad.UnitArr(w, units="km/s"), v, theta, pygad.UnitArr(J_lc, units=J_unit), num_J_lc, bh_stars_J, num_vescs
    
    def _beta_r_helper(self):
        vspherical = spherical_components(self.main_snap.stars[self.galaxy_radius_mask]["pos"], self.main_snap.stars[self.galaxy_radius_mask]["vel"])
        beta_r, radbins, bincount = beta_profile(self.main_snap.stars[self.galaxy_radius_mask]["r"], vspherical, 0.05)
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
                s += f" > {k}\n"
        print(s)
    
    def make_hdf5(self, fname, exist_ok=False):
        if os.path.isfile(fname) and not exist_ok:
            raise ValueError("HDF5 file already exists!")
        with h5py.File(fname, mode="w") as f:
            #save the binary info
            bhb = f.create_group("bh_binary")
            data_list = ["bh_masses",
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

            #save the galaxy properties
            gp = f.create_group("galaxy_properties")
            data_list = ["relaxed_remnant_flag",
                "relaxed_stellar_velocity_dispersion",
                "relaxed_stellar_velocity_dispersion_projected",
                "relaxed_inner_DM_fraction",
                "virial_info",
                "relaxed_effective_radius",
                "relaxed_half_mass_radius",
                "radial_bin_centres",
                "relaxed_core_parameters",
                "relaxed_density_profile",
                "relaxed_density_profile_projected",
                "binding_energy_bins",
                "relaxed_triaxiality_parameters",
                "total_stellar_mass",
                "ifu_map_ah",
                "ifu_map_merger",
                "snapshot_times",
                "stellar_shell_inflow_velocity",
                "bh_binary_watershed_velocity",
                "beta_r",
                "ang_mom_diff_angle",
                "ang_mom",
                "loss_cone",
                "stars_in_loss_cone",
                "particle_count",
                "num_escaping_stars"]
            self._saver(gp, data_list)
            self._add_attr(f["/galaxy_properties/stellar_shell_inflow_velocity"], "shell_radius", self.shell_radius)
            self._add_attr(f["/galaxy_properties/binding_energy_bins"], "units", self._energy_units)

            #save the parent properties
            gP = f.create_group("parent_properties")
            self._saver(gP, ["parent_quantities"])

            #set up some meta data
            meta = f.create_group("meta")
            now = datetime.datetime.now()
            self._add_attr(meta, "merger_name", self.merger_name)
            self._add_attr(meta, "created", now.strftime(date_str))
            self._add_attr(meta, "created_by", username)
            self._add_attr(meta, "last_accessed", now.strftime(date_str))
            self._add_attr(meta, "last_user", username)
            meta.create_dataset("logs", data=self._log)

