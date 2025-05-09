from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import binned_statistic
from pygad import ExprMask
from baggins.env_config import _cmlogger
from baggins.cosmology import angular_scale
from baggins.mathematics import get_histogram_bin_centres

__all__ = [
    "MUSE_NFM",
    "MUSE_WFM",
    "HARMONI_SENSITIVE",
    "HARMONI_BALANCED",
    "HARMONI_SPATIAL",
    "Euclid_NISP",
    "Euclid_VIS",
    "ERIS_IFU",
    "JWST_IFU",
    "MICADO_WFM",
    "MICADO_NFM",
    "VLT_FORS2",
    "ERIS_NIX_NFM",
    "JWST_LSS",
]

_logger = _cmlogger.getChild(__name__)


class BasicInstrument(ABC):
    def __init__(self, fov, sampling, res=None, z=None):
        """
        Template class for defining basic observation instrument properties

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        sampling : float
            spatial sampling of instrument in arcsec/pixel
        res : float, optional
            angular resolution in arcsec, by default None
        z : float, optional
            redshift of observations, by default None
        """
        self.field_of_view = fov
        self.sampling = sampling
        if res is None:
            res = 1e-6
        self.angular_resolution = res
        self._ang_scale = None
        self.max_extent = 40.0
        if z is not None:
            self.redshift = z

    def _param_check(self):
        try:
            assert self._ang_scale is not None
        except AssertionError:
            _logger.exception("Redshift must be set first!", exc_info=True)
            raise RuntimeError

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, z):
        self._redshift = z
        self._ang_scale = angular_scale(z)

    @property
    def ang_scale(self):
        # in kpc/arcsec
        return self._ang_scale

    @property
    def pixel_width(self):
        # in kpc
        self._param_check()
        return self.sampling * self._ang_scale

    @property
    def resolution_kpc(self):
        self._param_check()
        return self.angular_resolution * self._ang_scale

    @property
    def extent(self):
        # in kpc
        self._param_check()
        return min(self._ang_scale * self.field_of_view, self.max_extent)

    @property
    def number_pixels(self):
        return int(self.extent / self.pixel_width)

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        return f'{self.name}:\n FoV: {self.field_of_view}"\n sampling: {self.sampling}"/pix\n angular resolution: {self.angular_resolution}"\n pixel width: {self.pixel_width:.3e}kpc\n # pixels: {self.number_pixels}\n extent: {self.extent}'


class MUSE_NFM(BasicInstrument):
    def __init__(self, z=None):
        """
        MUSE narrow field mode instrument. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
        """
        super().__init__(fov=7.42, sampling=0.025, res=0.2, z=z)


class MUSE_WFM(BasicInstrument):
    def __init__(self, z=None):
        """
        MUSE wide field mode instrument. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
        """
        super().__init__(fov=60, sampling=0.2, res=0.4, z=z)


class HARMONI_SENSITIVE(BasicInstrument):
    def __init__(self, z=None):
        """
        HARMONI optimised for sensitivity. Parameters taken from:
        https://elt.eso.org/instrument/HARMONI/
        """
        super().__init__(fov=3.04, sampling=20e-3, res=20e-3, z=z)


class HARMONI_BALANCED(BasicInstrument):
    def __init__(self, z=None):
        """
        HARMONI balanced for sensitivity and spatial. Parameters taken from:
        https://elt.eso.org/instrument/HARMONI/
        """
        super().__init__(fov=1.52, sampling=10e-3, res=20e-3, z=z)


class HARMONI_SPATIAL(BasicInstrument):
    def __init__(self, z=None):
        """
        HARMONI optimised for sensitivity. Parameters taken from:
        https://elt.eso.org/instrument/HARMONI/
        """
        super().__init__(fov=0.61, sampling=4e-3, res=20e-3, z=z)


class Euclid_NISP(BasicInstrument):
    def __init__(self, z=None):
        """ "
        "Euclid infrared bands. Parameters taken from:
        https://sci.esa.int/web/euclid/-/euclid-nisp-instrument
        """
        super().__init__(fov=0.722 * 3600, sampling=0.3, res=None, z=z)


class Euclid_VIS(BasicInstrument):
    def __init__(self, z=None):
        """
        "Euclid visible bands. Parameters taken from:
        https://sci.esa.int/web/euclid/-/euclid-vis-instrument
        """
        super().__init__(fov=0.709 * 3600, sampling=0.101, res=0.23, z=z)


class ERIS_IFU(BasicInstrument):
    def __init__(self, z=None):
        """
        Eris IFU. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/eris/doc/ERIS_User_Manual_v116.0.pdf
        """
        super().__init__(fov=0.8, sampling=25, res=0.1, z=z)


class JWST_IFU(BasicInstrument):
    def __init__(self, z=None):
        """
        JWST IFU. Parameters taken from:
        https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0
        """
        super().__init__(fov=3, sampling=0.1, res=None, z=z)


class LongSlitInstrument(BasicInstrument):
    @abstractmethod
    def __init__(
        self, fov, sampling, slit_width, slit_length, res=None, z=None, rng=None
    ):
        """
        Base class for long slit spectroscopy instruments.

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        sampling : float
            spatial sampling of instrument in arcsec/pixel
        slit_width : float
            width of slit in arcsecs
        slit_length : float
            length of slit in arcsecs
        res : float, optional
            angular resolution in arcsec, by default None
        """
        super().__init__(fov, sampling, res, z=z)
        self.slit_width = slit_width
        self.slit_length = slit_length
        if rng is None:
            self._rng = np.random.default_rng()

    def get_slit_mask(self, xaxis=0, yaxis=2):
        """
        Mask for those particles in the slit.

        Parameters
        ----------
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2

        Returns
        -------
        mask : ExprMask
            pygad mask to select those particles within the slit
        """
        sl_phys = self.slit_length * self._ang_scale
        if sl_phys > self.extent:
            _logger.warning(
                f"Truncating slit length ({sl_phys:.1e}) to maximum extent ({self.extent:.1e})!"
            )
            sl_phys = self.extent
        sw_phys = self.slit_width * self._ang_scale
        mask = ExprMask(f"abs(pos[:,{xaxis}]) <= {0.5 * sl_phys}") & ExprMask(
            f"abs(pos[:,{yaxis}]) <= {0.5 * sw_phys}"
        )
        return mask

    def get_LOS_velocity_dispersion_profile(
        self, snap, N=100, xaxis=0, yaxis=2, N_per_bin=100
    ):
        """
        Calculate a 1D velocity dispersion profile using the slit. Note that no
        centring is done.

        Parameters
        ----------
        snap : pygad.Snapshot
            snapshot to analyse
        N : int, optional
            number of pseudoparticles per particle to generate, by default 100
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2

        Returns
        -------
        : np.array
            centres of pixels that define the slit
        vel_disp : np.array
            velocity dispersion along the long side of the slit
        """
        LOS_axis = list(set({0, 1, 2}).difference({xaxis, yaxis}))[0]
        mask = self.get_slit_mask(xaxis=xaxis, yaxis=yaxis)
        x = np.array(
            [
                xx + self._rng.normal(0, self.resolution_kpc, size=N)
                for xx in snap.stars[mask]["pos"][:, xaxis]
            ]
        ).flatten()
        V = np.repeat(snap.stars[mask]["vel"][:, LOS_axis], N).flatten()
        bins = np.quantile(x, np.linspace(0, 1, int(len(x) / N_per_bin) + 1))
        _logger.debug(f"Starting with {len(bins)} bins for LSS")
        # if bin difference is less than instrument sampling, join bins
        if np.any(np.diff(bins) < self.pixel_width):
            # which bins are smaller than the pixel width
            _bins = np.full_like(bins, np.nan)
            _bins[0] = bins[0]
            offset = 0
            for i, b in enumerate(bins[1:]):
                if b - _bins[i - offset] > self.pixel_width:
                    _bins[i - offset + 1] = b
                else:
                    offset += 1
            bins = _bins[~np.isnan(_bins)]
        try:
            assert np.all(np.diff(bins) >= self.pixel_width)
        except AssertionError:
            _logger.exception(
                "Some slit bins are narrower than the instrument resolution!",
                exc_info=True,
            )
            raise
        _logger.debug(
            f"There are {len(bins)} bins for LSS. Minimum bin width is {np.min(np.diff(bins))} (pixel width is {self.pixel_width})."
        )
        vel_disp, *_ = binned_statistic(x, V, bins=bins, statistic="std")
        return get_histogram_bin_centres(bins), vel_disp


class MICADO_WFM(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        MICADO for ELT
        https://elt.eso.org/instrument/MICADO/
        """
        super().__init__(
            fov=50.5,
            sampling=4e-3,
            res=50e-6,
            slit_width=16e-3,
            slit_length=3,
            rng=rng,
            z=z,
        )


class MICADO_NFM(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        MICADO for ELT
        https://elt.eso.org/instrument/MICADO/
        """
        super().__init__(
            fov=18,
            sampling=1.5e-3,
            res=20e-3,
            slit_width=16e-3,
            slit_length=3,
            rng=rng,
            z=z,
        )


class VLT_FORS2(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        VLT FORS2 instrument for long slit spectroscopy
        https://www.eso.org/sci/facilities/paranal/instruments/fors/doc/VLT-MAN-ESO-13100-1543_P116.2.pdf
        """
        super().__init__(
            fov=7.1 * 60,
            sampling=0.125,
            slit_width=0.28,
            slit_length=6.8 * 60,
            res=None,
            rng=rng,
            z=z,
        )


class ERIS_NIX_NFM(LongSlitInstrument):
    def __init__(self, z=None, rng=None):
        """
        Eris long slit. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/eris/doc/ERIS_User_Manual_v116.0.pdf
        """
        super().__init__(
            fov=26,
            sampling=13e-3,
            slit_width=68e-3,
            slit_length=12,
            res=0.1,
            z=z,
            rng=rng,
        )


class JWST_LSS(LongSlitInstrument):
    def __init__(self, z=None, rng=None):
        """
        JWST long slit. Parameters taken from:
        https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0
        """
        super().__init__(
            fov=3.4 * 60, sampling=0.1, slit_width=0.2, slit_length=3.2, z=z, rng=rng
        )
