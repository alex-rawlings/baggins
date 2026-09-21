import os
import shutil
from datetime import datetime
import h5py
import numpy as np
import pygad
import merger_ic_generator as mg
from baggins.initialise.ic_helpers import e_from_rperi
from baggins.env_config import _cmlogger, date_format
from baggins.utils import (
    read_parameters,
    write_calculated_parameters,
)
from baggins.analysis import (
    basic_snapshot_centring,
)
from baggins.mathematics import radial_separation

__all__ = ["MergerIC", "CircularInfallSystem"]

_logger = _cmlogger.getChild(__name__)


class MergerIC:
    def __init__(self, paramfile, exist_ok=False) -> None:
        """
        Class to initialise and edit Gadget merger simulations

        Parameters
        ----------
        paramfile : str
            path to .yml configuration file
        rng : numpy.random._generator.Generator, optional
            random number generator, by default None
        exist_ok : bool, optional
            allow overwriting of existing directories, by default False
        """
        self.paramfile = paramfile
        self.parameters = read_parameters(self.paramfile)
        if self.parameters["general"]["random_seed"] is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.parameters["general"]["random_seed"])
        self.exist_ok = exist_ok
        self._calc_quants = {}
        try:
            self.save_location = self.parameters["calculated"]["full_save_location"]
        except KeyError:
            self.save_location = None
        self._ic_file_names = []

    def _make_saveloc(self):
        """
        Convenience method to construct save location from parameter file

        Returns
        -------
        : str
            save location
        """
        self._calc_quants["full_save_location"] = os.path.join(
            self.parameters["file_locations"]["save_location"],
            f"{self.parameters['general']['galaxy_name_1']}-{self.parameters['general']['galaxy_name_2']}-{self._calc_quants['a0_physical']:.3f}-{self._calc_quants['e0']:.3f}",
        )
        return self._calc_quants["full_save_location"]

    def write_calculated_parameters(self):
        """
        Write calculated parameters to the parameter file
        """
        now = datetime.now()
        self._calc_quants["last_update"] = now.strftime(date_format)
        write_calculated_parameters(self._calc_quants, self.paramfile)

    def _calculate_input_orbital_quantities(self):
        oppars = self.parameters["orbital_properties"]

        # determine the radial units
        def _get_virial_radius():
            vr_list = []
            for i in range(1, 3):
                snap = pygad.Snapshot(
                    self.parameters["file_locations"][f"galaxy_file_{i}"], physical=True
                )
                basic_snapshot_centring(snap)
                vr, *_ = pygad.analysis.virial_info(snap)
                vr_list.append(vr)
            return float(max(vr_list))

        self._calc_quants["virial_radius_large"] = _get_virial_radius()

        # determine mass resolution
        self._calc_quants["mass_resolution"] = {}
        for i in range(1, 3):
            with h5py.File(
                self.parameters["file_locations"][f"galaxy_file_{i}"], "r"
            ) as f:
                try:
                    mbh = min(f["/PartType5/Masses"][:])
                    for parttype in range(5):
                        try:
                            m = min(f[f"/PartType{parttype}/Masses"][:])
                            self._calc_quants["mass_resolution"][
                                mg.ParticleType(parttype).name
                            ] = m / mbh
                        except KeyError:
                            pass
                except KeyError:
                    # no BH in merger
                    pass

        # determine initial semimajor axis
        try:
            assert oppars["a0"]["unit"] in ("virial", "kpc")
        except AssertionError:
            _logger.exception(
                f"Initial semimajor axis unit {oppars['a0']['unit']} not allowed! Must be one of ['kpc', 'virial']",
                exc_info=True,
            )
            raise
        if oppars["a0"]["unit"] == "virial":
            self._calc_quants["a0_physical"] = (
                self._calc_quants["virial_radius_large"] * oppars["a0"]["value"]
            )
        else:
            self._calc_quants["a0_physical"] = oppars["a0"]["value"]
        oppars["a0"] = self._calc_quants["a0_physical"]

        # determine eccentricity
        if oppars["e0"] is None:
            raise NotImplementedError
            _logger.info("Initial orbital eccentricity set from pericentre distance")
            self._calc_quants["e0"] = e_from_rperi(
                self._calc_quants["rperi_physical"]
                / self._calc_quants["virial_radius_large"]
            )
        else:
            self._calc_quants["e0"] = oppars["e0"]
        return oppars

    def _calculate_derived_orbital_quantities(self, merger):
        """
        Calculate quantities that are derived after the merger has been set up.

        Parameters
        ----------
        merger : mg.Merger
            merger system object
        """
        self._calc_quants["initial_BH_separation"] = radial_separation(
            merger.x1, merger.x2
        )[0]

        self._calc_quants["time_to_pericentre"] = merger.time_to_pericenter
        # print some velocity information about merger
        self._calc_quants["initial_velocity"] = {}
        for k in ("tangential", "radial"):
            self._calc_quants["initial_velocity"][k] = merger.initial_velocities[k]

    def _save_system(self, merger):
        """
        Save the merger system, and copy the parameter file to the output directory.

        Parameters
        ----------
        merger : mg.Merger
            merger system object
        """
        if self.save_location is None:
            self.save_location = self._make_saveloc()
            os.makedirs(self.save_location, exist_ok=self.exist_ok)
        file_name = os.path.join(
            self.save_location,
            f"{self.parameters['general']['galaxy_name_1']}-{self.parameters['general']['galaxy_name_2']}-{self._calc_quants['a0_physical']:.3f}-{self._calc_quants['e0']:.3f}.hdf5",
        )
        try:
            assert self.exist_ok or not os.path.exists(file_name)
        except AssertionError:
            _logger.exception(f"File {file_name} already exists!", exc_info=True)
            raise
        mg.write_hdf5_ic_file(
            filename=file_name,
            system=merger,
            center_CoM=self.parameters["general"]["recentre_merger_to_com"],
        )
        _logger.info(f"Merger IC file written to {file_name}")

        # copy parameter file to simulation directory
        shutil.copyfile(
            self.paramfile,
            os.path.join(self.save_location, os.path.basename(self.paramfile)),
        )
        # save parameters
        self.write_calculated_parameters()

    def generate_merger(self):
        """
        Set up a new merger system.

        Raises
        ------
        NotImplementedError
            for units other than 'virial' and 'kpc'
        """
        galaxy1 = mg.SnapshotSystem(
            self.parameters["file_locations"]["galaxy_file_1"],
            self.parameters["general"]["recentre_progens_to_com"],
        )
        galaxy2 = mg.SnapshotSystem(
            self.parameters["file_locations"]["galaxy_file_2"],
            self.parameters["general"]["recentre_progens_to_com"],
        )
        oppars = self._calculate_input_orbital_quantities()

        # edit oppars in place so we can pass it to Merger
        _oppars = {}
        for k in oppars.keys():
            _oppars[k.rstrip("0")] = oppars[k]

        merger = mg.Merger(galaxy1, galaxy2, **_oppars)

        self._calculate_derived_orbital_quantities(merger)
        self._save_system(merger)


class CircularInfallSystem(MergerIC):
    def __init__(self, paramfile, exist_ok=False):
        """
        Set up a satellite system on a circular orbit with respect to the host. To this end, in the paramfile, the first system is treated as the host, and the second as the satellite.

        Parameters are per MergerIC.
        """
        super().__init__(paramfile, exist_ok)

    def _calculate_derived_orbital_quantities(self, merger, r0):
        """
        Calculate quantities that are derived after the merger has been set up.

        Parameters
        ----------
        merger : mg.Merger
            merger system object
        r0 : float
            initial separation
        """
        self._calc_quants["initial_BH_separation"] = radial_separation(
            merger.x1, merger.x2
        )[0]
        # set m2 to 0 as we are setting the tangential velocity to be the
        # circular velocity of the host system
        orbit = mg.general.elliptic_orbit(
            merger.system1.total_enclosed_mass(r0), 0, a=r0, e=0
        )
        self._calc_quants["time_to_pericentre"] = abs(orbit["time_of_flight"])
        self._calc_quants["initial_velocity"] = {}
        for k, k2 in zip(("tangential", "radial"), ("v0t", "v0r")):
            self._calc_quants["initial_velocity"][k] = orbit[k2]

    def generate_merger(self):
        """
        Set up a new merger system.

        Raises
        ------
        NotImplementedError
            for units other than 'virial' and 'kpc'
        """
        system1 = mg.SnapshotSystem(
            self.parameters["file_locations"]["galaxy_file_1"],
            self.parameters["general"]["recentre_progens_to_com"],
        )
        system2 = mg.SnapshotSystem(
            self.parameters["file_locations"]["galaxy_file_2"],
            self.parameters["general"]["recentre_progens_to_com"],
        )
        r0 = self._calculate_input_orbital_quantities()["a0"]

        circ_vel = np.sqrt(mg.G * system1.total_enclosed_mass(r0) / r0)

        merger = mg.TwoBodySystem(
            system1,
            system2,
            x1=[0, 0, 0],
            x2=[r0, 0, 0],
            v1=[0, 0, 0],
            v2=[0, 0, circ_vel],
        )

        self._calculate_derived_orbital_quantities(merger, r0)
        self._save_system(merger)
