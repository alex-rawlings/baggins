from . import HDF5Base
from ...env_config import _cmlogger
from ...mathematics import radial_separation

__all__ = ["BrownianData"]

_logger = _cmlogger.copy(__file__)

class BrownianData(HDF5Base):
    def __init__(self) -> None:
        super().__init__()
        self._data_loaded = False
    
    # define properties
    @property
    def number_snaps(self):
        return self._number_snaps
    
    @property
    def bh1(self):
        return self._bh1

    @bh1.setter
    def bh1(self, v):
        self._bh_set_validator(v)
        self._bh1 = v

    @property
    def bh2(self):
        return self._bh2

    @bh2.setter
    def bh2(self, v):
        self._bh_set_validator(v)
        self._bh2 = v
    
    @property
    def radial_separation(self):
        return self._radial_separation

    @radial_separation.setter
    def radial_separation(self, v):
        self._radial_separation = v
    
    @property
    def snap_times(self):
        return self._snap_times

    @snap_times.setter
    def snap_times(self, v):
        self._snap_times = v
    

    # helper methods
    def _bh_set_validator(self, v):
        try:
            assert isinstance(v, dict)
        except AssertionError:
            _logger.logger.exception(f"BH Data must be of type 'dict'", exc_info=True)
        try:
            for k in ("id", "xoffset", "voffset"):
                assert k in v
        except AssertionError:
            _logger.logger.exception(f"BH Data must have key {k}", exc_info=True)

    def _offset_mag_helper(self, k):
        try:
            assert self._data_loaded
        except AssertionError:
            _logger.logger.exception(f"Data must be loaded before operations are performed on it!", exc_info=True)
            raise
        return radial_separation(self.bh1[k]), radial_separation(self.bh2[k])

    # public methods
    def compute_pos_offset_magnitude(self):
        return self._offset_mag_helper("xoffset")
    
    def compute_vel_offset_magnitude(self):
        return self._offset_mag_helper("voffset")
    
    def plot_time_series(self):
        raise NotImplementedError

    def plot_projection(self):
        raise NotImplementedError
    
    def load_from_file(self, f):
        raise NotImplementedError
    
    def print(self):
        raise NotImplementedError

        