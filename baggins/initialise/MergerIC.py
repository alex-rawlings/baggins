import os
import shutil
import re
from datetime import datetime
import h5py
import numpy as np
import pygad
import merger_ic_generator as mg

from initialise.mergers import e_from_rperi
from env_config import _cmlogger, date_format
from utils import read_parameters, write_calculated_parameters, get_snapshots_in_dir
from analysis import (
    get_com_of_each_galaxy,
    get_com_velocity_of_each_galaxy,
    get_virial_info_of_each_galaxy,
)
from general import snap_num_for_time
from mathematics import radial_separation
from analysis.masks import get_all_id_masks

__all__ = ["MergerIC"]

_logger = _cmlogger.getChild(__name__)


class MergerIC:
    def __init__(self, paramfile, rng=None, exist_ok=False) -> None:
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
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        self.exist_ok = exist_ok
        self._snaplist = None
        self._calc_quants = {}
        try:
            self.save_location = self.parameters["calculated"]["full_save_location"]
        except KeyError:
            self.save_location = None
        self.perturb_directories = []
        self._ic_file_names = []

    @property
    def snaplist(self):
        return self._snaplist

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
            f"{self.parameters['general']['galaxy_name_1']}-{self.parameters['general']['galaxy_name_2']}-{self._calc_quants['r0_physical']:.3f}-{self._calc_quants['rperi_physical']:.3f}",
        )
        return self._calc_quants["full_save_location"]

    def write_calculated_parameters(self):
        """
        Write calculated parameters to the parameter file
        """
        now = datetime.now()
        self._calc_quants["last_update"] = now.strftime(date_format)
        write_calculated_parameters(self._calc_quants, self.paramfile)

    def find_snapfile_to_perturb(self):
        """
        Determine the snapshot to perturb

        Returns
        -------
        snapfile : str
            path to snapshot that is closest to the desired perturbing time
        """
        ppars = self.parameters["perturb_properties"]
        # find the snapshot corresponding to the time we want
        self._snaplist = get_snapshots_in_dir(
            os.path.join(self.save_location, "output")
        )
        self._calc_quants["perturb_snap_idx"] = snap_num_for_time(
            self.snaplist,
            ppars["perturb_time"]["value"],
            units=ppars["perturb_time"]["unit"],
        )
        snapfile = self.snaplist[self._calc_quants["perturb_snap_idx"]]
        snap = pygad.Snapshot(snapfile, physical=True)
        bhsep = pygad.utils.geo.dist(snap.bh["pos"][0, :], snap.bh["pos"][1, :])
        _logger.info(f"BH separation when perturbed: {bhsep[0]:.2f} {bhsep.units}")
        try:
            assert bhsep > ppars["perturb_bhs"]["perturb_position"]["value"]
        except AssertionError:
            _logger.exception(
                f"BH separation {bhsep[0]:.2f} is less than the perturbation scale {ppars['perturb_bhs']['perturb_position']['value']:.2f}!",
                exc_info=True,
            )
            raise
        return snapfile

    def setup(self):
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
        oppars = self.parameters["orbital_properties"]

        # determine the radial units
        def _get_virial_radius():
            vr_list = []
            for i in [1, 2]:
                snap = pygad.Snapshot(
                    self.parameters["file_locations"][f"galaxy_file_{i}"], physical=True
                )
                xcom = get_com_of_each_galaxy(snap, method="ss", family="stars")
                vr, *_ = get_virial_info_of_each_galaxy(snap, xcom=xcom)
                vr_list.append(vr)
            return float(max(vr_list))

        # determine mass radius "fudge" factor
        if oppars["mass_radius_fac"] is None:
            oppars["mass_radius_fac"] = 0.5
            _logger.warning(
                f"Setting merger `mass_radius_fac` to default value of {oppars['mass_radius_fac']}"
            )

        self._calc_quants["virial_radius_large"] = _get_virial_radius()

        # determine initial separation
        try:
            assert oppars["r0"]["unit"] in ("virial", "kpc")
        except AssertionError:
            _logger.exception(
                f"Initial separation unit {oppars['r0']['unit']} not allowed! Must be one of ['kpc', 'virial']",
                exc_info=True,
            )
            raise
        if oppars["r0"]["unit"] == "virial":
            self._calc_quants["r0_physical"] = (
                self._calc_quants["virial_radius_large"] * oppars["r0"]["value"]
            )
        else:
            self._calc_quants["r0_physical"] = oppars["r0"]["value"]

        # determine first pericentre distance
        if oppars["rperi"]["value"] is not None:
            try:
                assert oppars["rperi"]["unit"] in ("virial", "kpc")
            except AssertionError:
                _logger.exception(
                    f"Pericentre distance unit {oppars['rperi']['unit']} not allowed! Must be one of ['kpc', 'virial']",
                    exc_info=True,
                )
                raise
            if oppars["rperi"]["unit"] == "virial":
                self._calc_quants["rperi_physical"] = (
                    self._calc_quants["virial_radius_large"] * oppars["rperi"]["value"]
                )
            else:
                self._calc_quants["rperi_physical"] = oppars["rperi"]["value"]
        else:
            try:
                assert oppars["a0"]["value"] is not None
                assert oppars["a0"]["unit"] in ("virial", "kpc")
                assert oppars["e0"] is not None
            except AssertionError:
                _logger.exception(
                    "'a0' (in units of virial or kpc) and 'e0' must be specified if 'rperi' is not!",
                    exc_info=True,
                )
                raise
            _logger.info(
                "Determining 'rperi' from initial semimajor axis and eccentricity"
            )
            if oppars["a0"]["unit"] == "virial":
                self._calc_quants["rperi_physical"] = (
                    self._calc_quants["virial_radius_large"]
                    * oppars["a0"]["value"]
                    * (1 - oppars["e0"])
                )
            else:
                self._calc_quants["rperi_physical"] = oppars["a0"]["value"] * (
                    1 - oppars["e0"]
                )

        # determine mass resolution
        mass_resolution = []
        for i in range(1, 3):
            with h5py.File(
                self.parameters["file_locations"][f"galaxy_file_{i}"], "r"
            ) as f:
                mstar = np.unique(f["/PartType4/Masses"][:])
                try:
                    assert len(mstar) == 1
                except AssertionError:
                    _logger.exception(
                        f"Non-unique stellar particle mass! {len(mstar)} masses present!",
                        exc_info=True,
                    )
                    raise
                mbh = min(f["/PartType5/Masses"][:])
                mass_resolution.append(mbh / mstar[0])
        self._calc_quants["mass_resolution"] = min(mass_resolution)

        # determine eccentricity
        if oppars["e0"] is None:
            _logger.info("Initial orbital eccentricity set from pericentre distance")
            self._calc_quants["e0"] = e_from_rperi(
                self._calc_quants["rperi_physical"]
                / self._calc_quants["virial_radius_large"]
            )
        else:
            self._calc_quants["e0"] = oppars["e0"]

        merger = mg.Merger(
            galaxy1,
            galaxy2,
            r0=self._calc_quants["r0_physical"],
            rperi=self._calc_quants["rperi_physical"],
            e=self._calc_quants["e0"],
            mass_radius_fac=oppars["mass_radius_fac"],
        )
        self._calc_quants["time_to_pericentre"] = merger.time_to_pericenter

        if self.save_location is None:
            self.save_location = self._make_saveloc()
        os.makedirs(os.path.join(self.save_location, "output"), exist_ok=self.exist_ok)
        file_name = os.path.join(
            self.save_location,
            f"{self.parameters['general']['galaxy_name_1']}-{self.parameters['general']['galaxy_name_2']}-{self._calc_quants['r0_physical']:.3f}-{self._calc_quants['rperi_physical']:.3f}.hdf5",
        )
        try:
            assert not os.path.exists(file_name)
        except AssertionError:
            _logger.exception(f"File {file_name} already exists!", exc_info=True)
            raise
        mg.write_hdf5_ic_file(
            filename=file_name,
            system=merger,
            save_plots=False,
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
        # print some velocity information about merger
        _logger.info("Initial merger velocities")
        for k in ("tangential", "radial"):
            _logger.info(f"- {k}: {merger.initial_velocities[k]:.3f} km/s")

    def create_perturbation_directories(self, file_to_copy, paramfile="paramfile"):
        """
        Create subdirectories and copy relevant files for a series of perturbed
        runs

        Parameters
        ----------
        file_to_copy : str, path-like
            snapshot file to copy as the new IC file
        paramfile : str, optional
            gadget parameter file, by default "paramfile"
        """
        ppars = self.parameters["perturb_properties"]
        perturb_dir = os.path.join(
            self.save_location, self.parameters["file_locations"]["perturb_sub_dir"]
        )
        os.makedirs(perturb_dir, exist_ok=self.exist_ok)
        for i in range(ppars["number_perturbs"]):
            _logger.info(f"Setting up child directory: {i}")
            child_dir = os.path.join(perturb_dir, f"{i:03d}")
            os.makedirs(os.path.join(child_dir, "output"), exist_ok=self.exist_ok)
            shutil.copyfile(
                os.path.join(self.save_location, paramfile),
                os.path.join(child_dir, paramfile),
            )
            self.perturb_directories.append(child_dir)
            self._ic_file_names.append(
                f"{self.parameters['general']['galaxy_name_1']}{self.parameters['general']['galaxy_name_2']}_perturb_{i:03d}"
            )
            shutil.copyfile(
                file_to_copy, os.path.join(child_dir, f"{self._ic_file_names[i]}.hdf5")
            )

    def update_gadget_paramfile(self, pfile, params):
        """
        Update a Gadget parameter file for a perturbed run

        Parameters
        ----------
        pfile : str
            path to gadget parameter file to edit
        params : dict
            parameters to update (keys: parameter name, value: new value)
        """
        with open(pfile, "r+") as f:
            contents = f.read()
            for param, val in params.items():
                line = re.search(
                    r"^\b{}\b.*".format(param), contents, flags=re.MULTILINE
                )
                if line is None:
                    _logger.warning(
                        f"Parameter {param} not in file! Parameter will not be updated."
                    )
                    continue
                if "%" in line.group(0):
                    comment = "  %" + "%".join(line.group(0).split("%")[1:])
                else:
                    comment = ""
                contents, numsubs = re.subn(
                    r"^\b{}\b.*".format(param),
                    "{}  {}{}".format(param, val, comment),
                    contents,
                    flags=re.MULTILINE,
                )
            f.seek(0)
            f.write(contents)
            f.truncate()

    def perturb_bhs(self):
        """
        Perturb the BHs of a merger system by a Gaussian distribution.
        XXX: For consistency with the large sample already run, the
        perturbation is applied along each coordinate axis. For future, it
        would be worth considering creating a perturbation of a given magnitude
        that is then projected along the different coordinate axes.
        """
        ppars = self.parameters["perturb_properties"]
        snapfile = self.find_snapfile_to_perturb()
        snap = pygad.Snapshot(snapfile, physical=True)
        # get com motions
        star_id_masks = get_all_id_masks(snap)
        xcoms = get_com_of_each_galaxy(
            snap, method="ss", masks=star_id_masks, family="stars"
        )
        vcoms = get_com_velocity_of_each_galaxy(snap, xcoms, masks=star_id_masks)

        # set up children directories and ICs
        self.create_perturbation_directories(snapfile)
        # for each perturbation 'child'
        for i, (child_dir, ic_name) in enumerate(
            zip(self.perturb_directories, self._ic_file_names)
        ):
            # edit BH coordinates
            fname = os.path.join(child_dir, f"{ic_name}.hdf5")
            _logger.debug(f"Perturbing file: {fname}")
            snap = pygad.Snapshot(fname, physical=True)
            for bhid in star_id_masks.keys():
                bhid_mask = bhid == snap.bh["ID"]
                _logger.debug(
                    f"Before perturb BH {bhid} has:\n position: {snap.bh['pos'][bhid_mask]}\n velocity: {snap.bh['vel'][bhid_mask]}"
                )
                snap.bh["pos"][bhid_mask] = pygad.UnitArr(
                    np.atleast_2d(
                        self.rng.normal(
                            xcoms[bhid],
                            ppars["perturb_bhs"]["perturb_position"]["value"],
                        )
                    ),
                    units=snap["pos"].units,
                )
                snap.bh["vel"][bhid_mask] = pygad.UnitArr(
                    np.atleast_2d(
                        self.rng.normal(
                            vcoms[bhid],
                            ppars["perturb_bhs"]["perturb_velocity"]["value"],
                        )
                    ),
                    units=snap["vel"].units,
                )
                _logger.debug(
                    f"After perturb BH {bhid} has:\n position: {snap.bh['pos'][bhid_mask]}\n velocity: {snap.bh['vel'][bhid_mask]}"
                )
            snap.write(fname, overwrite=True, gformat=3, double_prec=True)
            # add file names to update
            update_pars = ppars["gadget_parameters_to_update"]
            update_pars["InitCondFile"] = ic_name
            update_pars["SnapshotFileBase"] = ic_name
            # edit paramfile
            gadget_file = os.path.join(child_dir, "paramfile")
            self.update_gadget_paramfile(gadget_file, update_pars)
            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap

        # add new parameters to file
        self.write_calculated_parameters()
        _logger.info("All child directories made.")

    def perturb_field_particle(self):
        """
        Perturb a single field particle (star or DM) of a merger simulation
        """
        perturbation = self.parameters["perturb_properties"]["perturb_field_particles"][
            "perturb_position"
        ]["value"]
        radial_lims = self.parameters["perturb_properties"]["perturb_field_particles"][
            "radial_bounds"
        ]["value"]
        family = self.parameters["perturb_properties"]["perturb_field_particles"][
            "family"
        ]
        PartTypes = {"stars": "PartType4", "dm": "PartType1"}
        snapfile = self.find_snapfile_to_perturb()
        _logger.info(f"IC file: {snapfile}")
        assert radial_lims[0] < radial_lims[1]
        assert isinstance(perturbation, list) and len(perturbation) == 3
        # set up children directories and ICs
        self.create_perturbation_directories(snapfile)

        for i, (child_dir, ic_name) in enumerate(
            zip(self.perturb_directories, self._ic_file_names)
        ):
            this_snapfile = os.path.join(child_dir, f"{ic_name}.hdf5")
            with h5py.File(this_snapfile, "r+") as f:
                pos = f[f"/{PartTypes[family]}/Coordinates"][:]
                bh_pos_1 = f["/PartType5/Coordinates"][0, :]
                bh_pos_2 = f["/PartType5/Coordinates"][1, :]
                r1 = radial_separation(pos, bh_pos_1)
                r2 = radial_separation(pos, bh_pos_2)
                mask = np.logical_and(
                    np.logical_and(r1 < radial_lims[1], r1 > radial_lims[0]),
                    r2 > radial_lims[0],
                )
                try:
                    assert np.sum(mask) > 0
                except AssertionError:
                    _logger.exception(
                        f"There are no field particles of type {family} in the radial range {radial_lims[0]} - {radial_lims[1]}! Try expanding the radial range.",
                        exc_info=True,
                    )
                    raise

                selected_part_idx = self.rng.choice(np.arange(pos.shape[0])[mask])
                _logger.info(
                    f"Selected {family} ID: {f[f'/{PartTypes[family]}/ParticleIDs'][selected_part_idx]}"
                )
                _logger.debug("Distance to SMBHs: ")
                _logger.debug(f"  SMBH 1: {r1[selected_part_idx]}")
                _logger.debug(f"  SMBH 2: {r2[selected_part_idx]}")
                _logger.debug("Before perturbing: ")
                _logger.debug(
                    f[f"/{PartTypes[family]}/Coordinates"][selected_part_idx, :]
                )
                f[f"/{PartTypes[family]}/Coordinates"][
                    selected_part_idx, :
                ] += perturbation

            with h5py.File(this_snapfile, "r") as f:
                _logger.debug("After perturbing: ")
                _logger.debug(
                    f[f"/{PartTypes[family]}/Coordinates"][selected_part_idx, :]
                )

            update_pars = {"InitCondFile": ic_name, "SnapshotFileBase": ic_name}
            gadget_file = os.path.join(child_dir, "paramfile")
            self.update_gadget_paramfile(gadget_file, update_pars)
        self.write_calculated_parameters()
