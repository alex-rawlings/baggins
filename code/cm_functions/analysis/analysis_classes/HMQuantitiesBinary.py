import os.path
import numpy as np
import h5py
import ketjugw
import pygad

from . import HMQuantitiesBinaryData, HMQuantitiesSingle
from ..analyse_snap import (
    get_com_velocity_of_each_galaxy,
    influence_radius,
    hardening_radius,
    projected_quantities,
    get_com_of_each_galaxy,
    inner_DM_fraction,
    determine_if_merged,
    velocity_anisotropy,
    get_massive_bh_ID,
    get_G_rho_per_sigma,
)
from ..orbit import (
    get_bound_binary,
    get_binary_before_bound,
    move_to_centre_of_mass,
    find_pericentre_time,
    deflection_angle,
)
from ...env_config import _cmlogger, date_format, username
from ...general import convert_gadget_time, units
from ...mathematics import radial_separation
from ...utils import read_parameters, get_snapshots_in_dir, get_ketjubhs_in_dir


__all__ = ["HMQuantitiesBinary"]

_logger = _cmlogger.getChild(__name__)


class HMQuantitiesBinary(HMQuantitiesBinaryData, HMQuantitiesSingle):
    def __init__(self, parameter_file, merger_file, data_directory, merger_id) -> None:
        """
        Class to extract and save key quantities that may be useful for hierarchical modelling of a BH binary system.

        Parameters
        ----------
        parameter_file : str, path-like
            path to the analysis parameter file
        data_directory : str, path-like
            path to where the data is located
        merger_id : str
            ID tag for the merger family
        """
        HMQuantitiesBinaryData().__init__()
        HMQuantitiesSingle().__init__(
            parameter_file, merger_file, data_directory, merger_id
        )

        # predefined variables that we will loop through in the snapshots
        self.semimajor_axis_of_snapshot = []
        self.influence_radius = []
        self.hardening_radius = []
        self.G_rho_per_sigma = []

        ##------------------- Determine binary quantities -------------------##

        bh1, bh2, merged = get_bound_binary(self.ketju_file)
        orbit_pars = ketjugw.orbital_parameters(bh1, bh2)
        bh1_pb, bh2_pb, bound_state = get_binary_before_bound(self.ketju_file)

        # time that binary is bound
        self.binary_time = orbit_pars["t"] / units.Myr

        # semimajor axis of binary
        self.semimajor_axis = orbit_pars["a_R"] / units.kpc

        # eccentricity of binary
        self.eccentricity = orbit_pars["e_t"]

        # angular momentum of binary
        self.binary_angular_momentum = radial_separation(
            ketjugw.orbital_angular_momentum(bh1, bh2)
        )

        # energy of binary
        self.binary_energy = ketjugw.orbital_energy(bh1, bh2)

        # radial separation of binary
        self.binary_separation = radial_separation(bh1.x / units.kpc, bh2.x / units.kpc)

        # period of binary
        self.binary_period = 2 * np.pi / orbit_pars["n"] / units.Myr

        # masses of BHs
        self.binary_masses = [bh1.m[0], bh2.m[0]]

        # pericentre deflection angle before binary is bound
        bh1_pb, bh2_pb = move_to_centre_of_mass(bh1_pb, bh2_pb)
        try:
            _, peri_idxs = find_pericentre_time(bh1_pb, bh2_pb, prominence=0.005)
            self.prebound_deflection_angles = deflection_angle(
                bh1_pb, bh2_pb, peri_idxs
            )
        except:
            _logger.exception(
                f"Unable to determine pericentre times before binary is bound!",
                exc_info=True,
            )
            self.prebound_deflection_angles = []

    def calculate(self):
        """
        Calculate the quantities from the input snapshot(s)
        """
        N = len(self.snaplist)
        for snapfile in self.snaplist:
            snap = pygad.Snapshot(snapfile, physical=True)
            if not self._masses_set:
                self._set_masses(snap)
            t = convert_gadget_time(snap, new_unit="Myr")
            if t < self.binary_time[0]:
                # snapshot is from before binary is bound, let's skip
                _logger.debug(
                    f"Snapshot {snapfile} is before the binary is bound --> skipping."
                )
                snap.delete_blocks()
                pygad.gc_full_collect()
                del snap
                N -= 1
                continue
            self._snapshot_quantities_single(snap, t)

            # semimajor axis of binary at time of snapshot
            if not determine_if_merged(snap)[0]:
                idx = self.get_idx_in_vec(t, self.binary_time)
                self.semimajor_axis_of_snapshot.append(self.semimajor_axis[idx])
            else:
                self.semimajor_axis_of_snapshot.append(np.array([np.nan]))

            # some important radii
            self.influence_radius.append(max(list(influence_radius(snap).values())))
            try:
                self.hardening_radius.append(
                    hardening_radius(snap.bh["mass"], self.influence_radius[-1])
                )
            except AssertionError:
                self.hardening_radius.append(
                    pygad.UnitScalar(np.nan, self.influence_radius[-1].units)
                )

            # inner density and dispersion
            # method interpolates between snapshot i and snapshot i+1: protect
            # against the case of the final snapshot
            if self._snap_counter < N - 1:
                self.G_rho_per_sigma.append(
                    get_G_rho_per_sigma(self.snaplist, t, self.influence_radius[-1])
                )

            # clean up for next iteration
            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap
            # increment counter
            self._snap_counter += 1
            _logger.debug(f"Analysed {self._snap_counter} from {N} snapshots.")

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
                _logger.exception("HDF5 file already exists!", exc_info=True)
                raise

        with h5py.File(fname, mode="w") as f:
            self._hdf5_save_helper(f, fname)

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
                "prebound_deflection_angles",
            ]
            self._saver(bhb, _bhb_dl)
            _logger.info(f"BH binary quantities saved to {fname}")

            gpb = f.create_group("galaxy_binary_properties")
            _gpb_dl = [
                "semimajor_axis_of_snapshot",
                "influence_radius",
                "hardening_radius",
                "G_rho_per_sigma",
            ]
            self._saver(gpb, _gpb_dl)
            _logger.info(f"Galaxy binary property quantities saved to {fname}")

    @classmethod
    def load_from_file(self, f):
        try:
            raise NotImplementedError
        except NotImplementedError:
            _logger.exception(
                f"Class {self.__name__} does not inherit the method <load_from_file> from {self.__base__.__name__}",
                exc_info=True,
            )
            raise
