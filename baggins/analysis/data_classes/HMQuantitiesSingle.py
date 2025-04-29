import copy
import os.path
import numpy as np
import datetime
import h5py
import pygad
from baggins.analysis.data_classes.HMQuantitiesSingleData import (
    HMQuantitiesSingleData,
)
from baggins.analysis.analyse_snap import (
    get_com_of_each_galaxy,
    get_com_velocity_of_each_galaxy,
    get_massive_bh_ID,
    projected_quantities,
    inner_DM_fraction,
    velocity_anisotropy,
    escape_velocity,
)
from baggins.analysis.analyse_ketju import get_bh_particles
from baggins.env_config import _cmlogger, date_format, username
from baggins.general import convert_gadget_time
from baggins.utils import get_ketjubhs_in_dir, get_snapshots_in_dir, read_parameters


__all__ = ["HMQuantitiesSingle"]

_logger = _cmlogger.getChild(__name__)


class HMQuantitiesSingle(HMQuantitiesSingleData):
    def __init__(
        self, parameter_file, merger_file, data_directory, merger_id, snaps=None
    ) -> None:
        """
        Class to extract and save key quantities that may be useful for hierarchical modelling of a general galaxy system. Note that BH binary
        information is not analysed.

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
        try:
            assert len(kf) == 1
        except AssertionError:
            error_str = "Multiple" if len(kf) > 1 else "No"
            _logger.exception(
                f"{error_str} Ketju BH files found in directory {self.data_directory}. Only one file may be used to create a HMQuantitiesBinary object.",
                exc_info=True,
            )
            raise
        self.ketju_file = kf[0]
        self.snaplist = get_snapshots_in_dir(self.data_directory)
        if snaps is not None:
            try:
                assert isinstance(snaps, list)
                self.snaplist = [self.snaplist[i] for i in snaps]
            except AssertionError:
                _logger.exception(
                    f"Selecting specific snapshots requires `snaps` to be a list, not type <{type(snaps)}>",
                    exc_info=True,
                )
                raise
        self._analysis_params = read_parameters(parameter_file)
        self._merger_params = read_parameters(merger_file)
        self.radial_edges = copy.copy(
            self._analysis_params["galaxy"]["radial_edges"]["value"]
        )
        self.initial_galaxy_orbit = {}
        for k in ("e0", "r0_physical", "rperi_physical"):
            self.initial_galaxy_orbit[k] = copy.copy(
                self._merger_params["calculated"][k]
            )

        # predefined variables that we will loop through in snapshots
        self.analysed_snapshots = []
        self.time_of_snapshot = []
        self.virial_mass = []
        self.virial_radius = []
        self.half_mass_radius = []
        self.projected_mass_density = {}
        self.effective_radius = {}
        self.vel_dispersion_1Re_2 = {}
        self.projected_vel_dispersion_2 = {}
        self.inner_DM_fraction = {}
        self.velocity_anisotropy = {}
        self.escape_velocity = {}
        self.masses_in_galaxy_radius = {"stars": [], "dm": [], "bh": []}
        self.particle_masses = {"stars": None, "dm": None, "bh": None}

        # some helper variables
        self._has_dm = True
        self._masses_set = False
        self._snap_counter = 0

        # #------------------- Determine merger properties -------------------##
        bh1, bh2, merged = get_bh_particles(self.ketju_file)
        self.merger_remnant = {
            "merged": False,
            "mass": None,
            "spin": None,
            "kick": None,
        }
        if merged():
            self.merger_remnant["merged"] = True
            self.merger_remnant["mass"] = merged.mass
            self.merger_remnant["spin"] = merged.chi
            self.merger_remnant["kick"] = merged.kick_magnitude

    @property
    def snaplist(self):
        return self._snaplist

    @snaplist.setter
    def snaplist(self, v):
        self._snaplist = v

    # function to set particle masses
    def _set_masses(self, s):
        """
        Helper function to set the particle masses.

        Parameters
        ----------
        s : pygad.Snapshot
            snapshot to analyse
        """
        self.particle_masses["stars"] = max(np.unique(s.stars["mass"]))
        try:
            self.particle_masses["dm"] = max(np.unique(s.dm["mass"]))
        except ValueError:
            _logger.warning("DM particles do not exist for this run")
            self._has_dm = False
        self.particle_masses["bh"] = min(np.unique(s.bh["mass"]))
        self._masses_set = True

    def _snapshot_quantities_single(self, s, t):
        """
        Helper function to facilitate looping.

        Parameters
        ----------
        s : pygad.Snapshot
            snapshot to analyse
        t : float, pygad.UnitQty
            time of snapshot
        """
        self.analysed_snapshots.append(s.filename)
        _logger.debug(f"Reading: {s.filename}")
        # get the centre
        _xcom = get_com_of_each_galaxy(s, method="ss", family="stars")
        bh_id = get_massive_bh_ID(s.bh)
        xcom = _xcom[bh_id]
        vcom = get_com_velocity_of_each_galaxy(s, _xcom)[bh_id]

        # aperture mask of galaxy_radius
        ball_mask = pygad.BallMask(
            pygad.UnitScalar(
                self._analysis_params["galaxy"]["maximum_radius"]["value"],
                self._analysis_params["galaxy"]["maximum_radius"]["unit"],
                subs=s,
            ),
            center=xcom,
        )

        # snapshot time
        self.time_of_snapshot.append(t)

        # virial info
        _vr, _vm = pygad.analysis.virial_info(s, center=xcom)
        self.virial_mass.append(_vm)
        self.virial_radius.append(_vr)

        # half mass radius
        self.half_mass_radius.append(
            pygad.analysis.half_mass_radius(s.stars[ball_mask], center=xcom)
        )

        # projected quantities
        k = f"{t:.3f}"
        _Re, _vsig2Re, _vsig2r, _Sigma = projected_quantities(
            s[ball_mask],
            obs=self._analysis_params["galaxy"]["num_projection_rotations"],
            r_edges=self.radial_edges,
        )
        self.effective_radius[k] = list(_Re.values())[0]
        self.vel_dispersion_1Re_2[k] = list(_vsig2Re.values())[0]
        self.projected_vel_dispersion_2[k] = list(_vsig2r.values())[0]
        self.projected_mass_density[k] = list(_Sigma.values())[0]

        # inner DM fraction
        self.inner_DM_fraction[k] = []
        if self._has_dm:
            for j, _re in enumerate(self.effective_radius[k]):
                self.inner_DM_fraction[k].append(
                    inner_DM_fraction(s[ball_mask], Re=_re, centre=xcom)
                )

        # beta profile
        self.velocity_anisotropy[k], *_ = velocity_anisotropy(
            s.stars[ball_mask], r_edges=self.radial_edges, xcom=xcom, vcom=vcom
        )

        # escape velocity as a function of raduius
        self.escape_velocity[k] = escape_velocity(s)(self.radial_edges)

        # masses
        self.masses_in_galaxy_radius["stars"].append(np.sum(s.stars[ball_mask]["mass"]))
        if self._has_dm:
            self.masses_in_galaxy_radius["dm"].append(np.sum(s.dm[ball_mask]["mass"]))
        self.masses_in_galaxy_radius["bh"].append(np.sum(s.bh[ball_mask]["mass"]))

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
            self._snapshot_quantities_single(snap, t)
            snap.delete_blocks()
            pygad.gc_full_collect()
            del snap
            # increment counter
            self._snap_counter += 1
            _logger.debug(f"Analysed {self._snap_counter} from {N} snapshots.")

    def _hdf5_save_helper(self, f, fname):
        """
        Helper function to facilitate saving.

        Parameters
        ----------
        f : TextIOWrapper
            handle to file to save data to
        fname : path-like
            file name
        """
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

        mgd = f.create_group("merged")
        _mgd_dl = ["merger_remnant"]
        self._saver(mgd, _mgd_dl)
        _logger.info(f"Merger remnant quantities saved to {fname}")

        gp = f.create_group("galaxy_properties")
        _gp_dl = [
            "analysed_snapshots",
            "radial_edges",
            "time_of_snapshot",
            "projected_mass_density",
            "projected_vel_dispersion_2",
            "effective_radius",
            "vel_dispersion_1Re_2",
            "half_mass_radius",
            "virial_mass",
            "virial_radius",
            "inner_DM_fraction",
            "velocity_anisotropy",
            "escape_velocity",
            "masses_in_galaxy_radius",
            "particle_masses",
            "initial_galaxy_orbit",
        ]
        self._saver(gp, _gp_dl)
        _logger.info(f"Galaxy property quantities saved to {fname}")

    def make_hdf5(self, fname, exist_ok=False):
        """
        Create a HDF5 file of the data that can then be read back into the
        python environment using the base class HMQuantitiesBinaryData.

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
