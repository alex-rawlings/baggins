import pygad
import ketjugw
from baggins.analysis.analysis_classes.BrownianData import BrownianData
from baggins.analysis.analyse_snap import (
    get_com_of_each_galaxy,
    get_com_velocity_of_each_galaxy,
)
from baggins.analysis.masks import get_all_id_masks
from baggins.analysis.analyse_ketju import get_bound_binary
from baggins.general.pygad_helper import convert_gadget_time, snap_num_for_time
from baggins.general.units import Myr
from baggins.utils import read_parameters, get_snapshots_in_dir, get_ketjubhs_in_dir
from baggins.env_config import _cmlogger

__all__ = ["Brownian"]

_logger = _cmlogger.getChild(__name__)


class Brownian(BrownianData):
    def __init__(self, analysis_parfile, data_dir) -> None:
        """
        Class to analyse Brownian motion of a series of snapshots.

        Parameters
        ----------
        analysis_parfile : str
            path to analysis .yml file
        data_dir : str
            path to snapshot data directory
        """
        super().__init__()
        self._analysis_pars = read_parameters(analysis_parfile)
        self.data_directory = data_dir
        self.snaplist = get_snapshots_in_dir(self.data_directory)
        kf = get_ketjubhs_in_dir(self.data_directory)
        try:
            assert len(kf) == 1
        except AssertionError:
            _logger.exception(
                f"Multiple Ketju BH files found in directory {self.data_directory}. Only one file may be used to create a HMQuantitiesBinary object."
            )
            raise
        self.ketju_file = kf[0]
        bh_dict = dict(
            id=None,
            xoffset=[],
            voffset=[],
        )
        self.bh1 = bh_dict
        self.bh2 = bh_dict
        self.snap_time = []
        self.radial_separation = []
        self._number_snaps = 0

        # collate Brownian motion for all snapshots until binary is bound
        _bh1, _bh2, *_ = get_bound_binary(self.ketju_file)
        orbit_pars = ketjugw.orbital_parameters(_bh1, _bh2)
        max_snap_idx = snap_num_for_time(
            self.snaplist, orbit_pars["t"] / Myr, method="floor"
        )

        for i, sf in enumerate(self.snaplist):
            if i > max_snap_idx:
                break
            snap = pygad.Snapshot(sf, physical=True)
            if i == 0:
                star_id_masks = get_all_id_masks(snap)
                self.bh1["id"], self.bh2["id"] = list(star_id_masks.keys())
            xcom = get_com_of_each_galaxy(
                snap, method="ss", masks=star_id_masks, family="stars"
            )
            vcom = get_com_velocity_of_each_galaxy(snap, xcom, masks=star_id_masks)
            for bh, x, v in zip(
                ((self.bh1, self.bh2)), list(xcom.values()), list(vcom.values())
            ):
                bh_id_mask = pygad.IDMask(bh["id"])
                bh["xoffset"].append(x - snap.bh[bh_id_mask]["pos"])
                bh["voffset"].append(v - snap.bh[bh_id_mask]["vel"])
            self._number_snaps += 1
            self.radial_separation.append(pygad.utils.geo.dist(snap.bh["pos"]))
            self.snap_time.append(convert_gadget_time(snap))

        self._data_loaded = True

    def save(self, fname):
        raise NotImplementedError
