import copy
import os.path
import numpy as np
import datetime
import h5py
import ketjugw
import pygad

from . import HMQuantitiesData
from ..analyse_snap import get_com_velocity_of_each_galaxy, influence_radius, hardening_radius, projected_quantities, get_com_of_each_galaxy, inner_DM_fraction, determine_if_merged, velocity_anisotropy, get_massive_bh_ID, get_G_rho_per_sigma
from ..orbit import get_bound_binary, get_binary_before_bound, move_to_centre_of_mass, find_pericentre_time, deflection_angle
from ...env_config import _cmlogger, date_format, username
from ...general import convert_gadget_time
from ...mathematics import radial_separation
from ...utils import read_parameters, get_snapshots_in_dir, get_ketjubhs_in_dir


__all__ = ["HMQuantities"]

_logger = _cmlogger.copy(__file__)

myr = ketjugw.units.yr * 1e6
kpc = ketjugw.units.pc * 1e3



class HMQuantities(HMQuantitiesData):
    def __init__(self, parameter_file, merger_file, data_directory, merger_id) -> None:
        """
        Class to extract and save key quantities that may be useful for hierarchical modelling.

        Parameters
        ----------
        parameter_file : str, path-like
            path to the analysis parameter file
        data_directory : str, path-like
            path to where the data is located
        merger_id : str
            ID tag for the merger family
        """
        super().__init__()
        self.data_directory = data_directory
        self.merger_id = merger_id
        kf = get_ketjubhs_in_dir(self.data_directory)
        # flag to skip DM methods if non-existent
        has_dm = True
        try:
            assert len(kf) == 1
        except AssertionError:
            _logger.logger.exception(f"Multiple Ketju BH files found in directory {self.data_directory}. Only one file may be used to create a HMQuantities object.", exc_info=True)
            raise
        self.ketju_file = kf[0]
        self.snaplist = get_snapshots_in_dir(self.data_directory)
        self._analysis_params = read_parameters(parameter_file)
        self._merger_params = read_parameters(merger_file)
        self.radial_edges = copy.copy(self._analysis_params["galaxy"]["radial_edges"]["value"])
        self.initial_galaxy_orbit = {}
        for k in ("e0", "r0_physical", "rperi_physical"):
            self.initial_galaxy_orbit[k] = copy.copy(self._merger_params["calculated"][k])

        ##------------------- Determine binary quantities -------------------##

        bh1, bh2, merged = get_bound_binary(self.ketju_file)
        orbit_pars = ketjugw.orbital_parameters(bh1, bh2)
        bh1_pb, bh2_pb = get_binary_before_bound(self.ketju_file)

        # time that binary is bound
        self.binary_time = orbit_pars["t"]/myr

        # semimajor axis of binary
        self.semimajor_axis = orbit_pars["a_R"] / kpc

        # eccentricity of binary
        self.eccentricity = orbit_pars["e_t"]

        # angular momentum of binary
        self.binary_angular_momentum = radial_separation(ketjugw.orbital_angular_momentum(bh1, bh2))

        # energy of binary
        self.binary_energy = ketjugw.orbital_energy(bh1, bh2)

        # radial separation of binary
        self.binary_separation = radial_separation(bh1.x / kpc, bh2.x / kpc)

        # period of binary
        self.binary_period = 2*np.pi / orbit_pars["n"] / myr

        # masses of BHs
        self.binary_masses = [bh1.m[0], bh2.m[0]]

        # pericentre deflection angle before binary is bound
        bh1_pb, bh2_pb = move_to_centre_of_mass(bh1_pb, bh2_pb)
        try:
            peri_idxs = find_pericentre_time(bh1_pb, bh2_pb, prominence=0.005)
            self.prebound_deflection_angles = deflection_angle(bh1_pb, bh2_pb, peri_idxs)
        except:
            _logger.logger.exception(f"Unable to determine pericentre times before binary is bound!", exc_info=True)
            self.prebound_deflection_angles = []

        ##------------------- Determine merger properties -------------------##

        self.merger_remnant = {"merged":False, "mass":None, "spin":None, "kick":None}
        if merged():
            self.merger_remnant["merged"] = True
            self.merger_remnant["mass"] = merged.mass
            self.merger_remnant["spin"] = merged.chi
            self.merger_remnant["kick"] = merged.kick_magnitude

        ##------------------- Determine galaxy quantities -------------------##

        # predefine variables
        self.analysed_snapshots = []
        self.time_of_snapshot = []
        self.semimajor_axis_of_snapshot = []
        self.influence_radius = []
        self.hardening_radius = []
        self.G_rho_per_sigma = []
        self.half_mass_radius = []
        self.virial_mass = []
        self.virial_radius = []
        self.effective_radius = {}
        self.vel_dispersion_1Re_2 = {}
        self.projected_vel_dispersion_2 = {}
        self.projected_mass_density = {}
        self.inner_DM_fraction = {}
        self.velocity_anisotropy = {}
        self.masses_in_galaxy_radius = {"stars":[], "dm":[], "bh":[]}
        self.particle_masses = {"stars":None, "dm":None, "bh":None}
        

        # loop through all snapshots
        # set counter here so that those rejected snapshots don't affect 
        # ordering
        i = 0
        N = len(self.snaplist)
        for snapfile in self.snaplist:
            snap = pygad.Snapshot(snapfile, physical=True)
            t = convert_gadget_time(snap, new_unit="Myr")
            if i==0:
                self.particle_masses["stars"] = max(np.unique(snap.stars["mass"]))
                try:
                    self.particle_masses["dm"] = max(np.unique(snap.dm["mass"]))
                except ValueError:
                    _logger.logger.warning("DM particles do not exist for this run")
                    has_dm = False
                self.particle_masses["bh"] = min(np.unique(snap.bh["mass"]))
            if t < self.binary_time[0]:
                # snapshot is from before binary is bound, let's skip
                _logger.logger.debug(f"Snapshot {snapfile} is before the binary is bound --> skipping.")
                snap.delete_blocks()
                pygad.gc_full_collect()
                del snap
                N -= 1
                continue
            self.analysed_snapshots.append(snapfile)
            _logger.logger.debug(f"Reading: {snapfile}")
            # get the centre
            _xcom = get_com_of_each_galaxy(snap, method="ss", family="stars")
            bh_id = get_massive_bh_ID(snap.bh)
            xcom = _xcom[bh_id]
            vcom = get_com_velocity_of_each_galaxy(snap, _xcom)[bh_id]

            # aperture mask of galaxy_radius
            ball_mask = pygad.BallMask(pygad.UnitScalar(self._analysis_params["galaxy"]["maximum_radius"]["value"], self._analysis_params["galaxy"]["maximum_radius"]["unit"], subs=snap), center=xcom)

            # snapshot time
            self.time_of_snapshot.append(t)

            # semimajor axis of binary at time of snapshot
            if not determine_if_merged(snap)[0]:
                idx = self.get_idx_in_vec(t, self.binary_time)
                self.semimajor_axis_of_snapshot.append(self.semimajor_axis[idx])
            else:
                self.semimajor_axis_of_snapshot.append(np.nan)

            # some important radii
            self.influence_radius.append(
                max(list(influence_radius(snap).values()))
            )
            self.hardening_radius.append(
                hardening_radius(snap.bh["mass"], self.influence_radius[i])
            )
            self.half_mass_radius.append(
                pygad.analysis.half_mass_radius(snap.stars[ball_mask], center=xcom)
            )

            # inner density and dispersion
            # method interpolates between snapshot i and snapshot i+1: protect 
            # against the case of the final snapshot
            if i<N-1:
                self.G_rho_per_sigma.append(
                    get_G_rho_per_sigma(self.snaplist, t, self.influence_radius[-1])
                )

            # virial info
            _vr, _vm = pygad.analysis.virial_info(snap, center=xcom)
            self.virial_mass.append(_vm)
            self.virial_radius.append(_vr)

            # projected quantities
            k = f"{t:.3f}"
            _Re, _vsig2Re, _vsig2r, _Sigma = projected_quantities(snap[ball_mask], obs=self._analysis_params["galaxy"]["num_projection_rotations"], r_edges=self.radial_edges)
            self.effective_radius[k] = list(_Re.values())[0]
            self.vel_dispersion_1Re_2[k] = list(_vsig2Re.values())[0]
            self.projected_vel_dispersion_2[k] = list(_vsig2r.values())[0]
            self.projected_mass_density[k] = list(_Sigma.values())[0]

            # inner DM fraction
            self.inner_DM_fraction[k] = []
            if has_dm:
                for j, _re in enumerate(self.effective_radius[k]):
                    self.inner_DM_fraction[k].append(
                        inner_DM_fraction(snap[ball_mask], Re=_re, centre=xcom)
                    )

            # beta profile
            self.velocity_anisotropy[k], *_ = velocity_anisotropy(snap.stars[ball_mask], r_edges=self.radial_edges, xcom=xcom, vcom=vcom)

            # masses
            self.masses_in_galaxy_radius["stars"].append(
                        np.sum(snap.stars[ball_mask]["mass"])
            )
            if has_dm:
                self.masses_in_galaxy_radius["dm"].append(
                            np.sum(snap.dm[ball_mask]["mass"])
                )
            self.masses_in_galaxy_radius["bh"].append(
                        np.sum(snap.bh[ball_mask]["mass"])
            )
            
            # clean up for next iteration
            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap
            # increment counter
            i += 1
            _logger.logger.debug(f"Analysed {i} from {N} snapshots.")


    @property
    def snaplist(self):
        return self._snaplist
    
    @snaplist.setter
    def snaplist(self, v):
        self._snaplist = v

    def make_hdf5(self, fname, exist_ok=False):
        """
        Create a HDF5 file of the data that can then be read back into the 
        python environment using the base class HMQuantitiesData.

        Parameters
        ----------
        fname : str, path-like
            filename to save to
        exist_ok : bool, optional
            allow overwriting of an existing file, by default False
        """
        if os.path.isfile(fname):
            try:
                assert exist_ok
            except AssertionError:
                _logger.logger.exception("HDF5 file already exists!", exc_info=True)
                raise
        
        with h5py.File(fname, mode="w") as f:
            # set up some meta data
            meta = f.create_group("meta")
            now = datetime.datetime.now()
            self._add_attr(meta, "data_directory", self.data_directory)
            self._add_attr(meta, "merger_id", self.merger_id)
            self._add_attr(meta, "created", now.strftime(date_format))
            self._add_attr(meta, "created_by", username)
            self._add_attr(meta, "last_accessed", now.strftime(date_format))
            self._add_attr(meta, "last_user", username)
            meta.create_dataset("logs", data=self._log)

            bhb = f.create_group("bh_binary")
            _bhb_dl = [
                        "binary_time",
                        "semimajor_axis",
                        "eccentricity",
                        "binary_angular_momentum",
                        "binary_energy",
                        "binary_separation",
                        "binary_period",
                        "binary_masses",
                        "prebound_deflection_angles"
            ]
            self._saver(bhb, _bhb_dl)
            _logger.logger.info(f"BH binary quantities saved to {fname}")

            mgd = f.create_group("merged")
            _mgd_dl = ["merger_remnant"]
            self._saver(mgd, _mgd_dl)
            _logger.logger.info(f"Merger remnant quantities saved to {fname}")

            gp = f.create_group("galaxy_properties")
            _gp_dl = [
                       "analysed_snapshots",
                       "radial_edges",
                       "time_of_snapshot",
                       "semimajor_axis_of_snapshot",
                       "influence_radius",
                       "hardening_radius",
                       "G_rho_per_sigma",
                       "projected_mass_density",
                       "projected_vel_dispersion_2",
                       "effective_radius",
                       "vel_dispersion_1Re_2",
                       "half_mass_radius",
                       "virial_mass",
                       "virial_radius",
                       "inner_DM_fraction",
                       "velocity_anisotropy",
                       "masses_in_galaxy_radius",
                       "particle_masses",
                       "initial_galaxy_orbit"
            ]
            self._saver(gp, _gp_dl)
            _logger.logger.info(f"Galaxy property quantities saved to {fname}")


