from abc import ABC
import os.path
import numpy as np
from copy import copy
import unyt
from synthesizer import grid, instruments
from astropy import cosmology
from baggins.env_config import _cmlogger, synthesizer_data
from baggins.utils import get_files_in_dir
from baggins.cosmology import angular_scale

__all__ = [
    "set_luminosity",
    "get_spectrum_ssp",
    "get_euclid_filter_collection",
    "get_hst_filter_collection",
    "get_surface_brightness",
    "get_flux_from_magnitude",
    "MUSE_NFM",
    "MUSE_WFM",
    "Euclid_NISP",
    "Euclid_VIS",
]


_logger = _cmlogger.getChild(__name__)


def set_luminosity(snap, sed, z=0):
    """
    Set the bolometric luminosity and magnitude for a gas-free snapshot in-place.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to set fields for
    metallicity : float, array-like
        metallicity values (must have same length as snap.stars if array)
    age : float, array-like
        age values (must have same length as snap.stars if array)
    z : float, optional
        redshift of source (for cosmological dimming), by default 0
    age_units : str, optional
        units for age, by default "Gyr"
    **kwargs
        other keyword-arguments parsed to pygad.get_luminosities()
    """
    try:
        assert set(snap.stars.all_blocks()).isdisjoint({"lum", "metallicity", "age"})
    except AssertionError:
        _logger.exception(
            "Cannot set blocks 'lum', 'metallicity', or 'age' to a custom value if they already exist!",
            exc_info=True,
        )
        raise
    # TODO calculate cosmological dimming here or elsewhere?
    # sed object doesn't store unyt conversions, so manually obtain the
    # conversion from erg/s to Lsol
    _sed = copy(sed)
    _sed.lnu *= snap.stars["mass"][0].view(np.ndarray)
    erg_per_s_per_Lsun = unyt.Lsun.get_conversion_factor(
        _sed.bolometric_luminosity.units
    )[0]
    snap.stars["lum"] = np.full(
        len(snap.stars),
        _sed.bolometric_luminosity.value / erg_per_s_per_Lsun / (1 + z) ** 4,
    )


def get_spectrum_ssp(
    age,
    metallicity,
    grid_name="bpass-2.2.1-bin_chabrier03-0.1,100.0.hdf5",
    grid_dir=None,
):
    """
    Get the spectrum of a population, given some age and metallicity. Assumes that all stellar mass contributes equally (i.e. no dust attenuation).

    Parameters
    ----------
    age : float
        age of a stellar particle (in Gyr)
    metallicity : float
        metallicity of stellar particle
    grid_name : str, optional
        SSP grid to query, by default "bpass-2.2.1-bin_chabrier03-0.1,100.0.hdf5"
    grid_dir : str, optional
        directory of grid_name, by default None

    Returns
    -------
    g : synthesizer.grid.Grid
        grid object
    sed : synthesizer.Sed
        spectral energy distribution object
    """
    if grid_dir is None:
        # use the default location for grids
        grid_dir = synthesizer_data
    try:
        assert os.path.isfile(os.path.join(grid_dir, grid_name))
    except AssertionError:
        valid_grids = get_files_in_dir(grid_dir, ext=".hdf5", name_only=True)
        _logger.exception(f"No grid called {grid_name}. Valid grids are {valid_grids}")
        raise
    _logger.info(f"Using data {os.path.join(grid_dir, grid_name)}")
    # create the grid
    g = grid.Grid(grid_name, grid_dir=grid_dir, read_lines=False)
    log10age = np.log10(age)
    # extract the spectrum at the target age / metallicity
    grid_point = g.get_grid_point(log10ages=log10age, metallicity=metallicity)
    sed = g.get_spectra(grid_point, spectra_id="incident")
    return g, sed


def get_euclid_filter_collection(g, new_lam_size=1000):
    """
    Convenience function to return all Euclid transmission filters.

    Parameters
    ----------
    g : synthesizer.grid.Grid
        grid object to query wavelengths from
    new_lam_size : int, optional
        resample grid with this many wavelength bins, by default 1000

    Returns
    -------
    euclid_filters : synthesizer.FilterCollection
        collection of Euclid filters
    """
    _filter_codes = ["Euclid/VIS.vis"]
    _filter_codes.extend([f"Euclid/NISP.{b}" for b in ("Y", "J", "H")])
    euclid_filters = instruments.FilterCollection(
        filter_codes=_filter_codes, new_lam=g.lam
    )
    euclid_filters.resample_filters(lam_size=new_lam_size)
    return euclid_filters


def get_hst_filter_collection(g, new_lam_size=1000):
    hst_filters = instruments.FilterCollection(
        filter_codes=["HST/ACS_HRC.F435W", "HST/ACS_HRC.F550W", "HST/ACS_HRC.F606W"],
        new_lam=g.lam,
    )
    hst_filters.resample_filters(lam_size=new_lam_size)
    return hst_filters


def get_surface_brightness(
    sed, stellar_mass, filters_collection, filter_code, z, pixel_size
):
    """
    Get the absolute and apparent magnitudes for an SED.

    Parameters
    ----------
    sed : synthesizer.Sed
        spectral energy distribution object
    stellar_mass : float
        total stellar mass in Msol
    filters_collection : synthesizer.FilterCollection
        collection of instrument filters
    filter_code : str
        specific filter we want to use
    z : float
        redshift of object
    pixel_size : float
        pixel side length (assumed square) in kpc

    Returns
    -------
    : dict
        absolute and apparent magnitudes
    """
    # need to copy the sed so we don't affect original
    _sed = copy(sed)
    # determine the angular scale
    inv_ang_scale = (
        cosmology.Planck18.arcsec_per_kpc_proper(z).value * unyt.arcsec / unyt.kpc
    )
    # "super-impose" spectra from all the stellar mass
    _sed.lnu *= stellar_mass
    _sed.get_fnu(cosmo=cosmology.Planck18, z=z)
    _flux = _sed.flux
    # convert from per kpc^2 to per arcsec^2
    _flux /= (pixel_size * unyt.kpc * inv_ang_scale) ** 2

    chosen_filter = filters_collection[filter_code]

    # normalising factor
    Fvo = 3631 * unyt.Jy
    # convert to erg/s/Hz/arcsec^2
    Fvo = Fvo.to("erg/(Hz*s*kpc**2)") / inv_ang_scale**2
    _logger.debug(f"Fvo is {Fvo}")

    # determine first apparent magnitude with transmission correction
    # and K correction
    r_per_10pc = (
        cosmology.Planck18.luminosity_distance(z).to("pc").value
        * unyt.pc
        / (10 * unyt.pc)
    )
    K_correction = -2.5 * np.log10(1 + z)
    app_mag = (
        -2.5 * np.log10(chosen_filter.apply_filter(_flux, nu=_sed.obsnu) / Fvo)
        + K_correction
    )

    # now convert to absolute magnitude
    abs_mag = app_mag - K_correction - r_per_10pc

    _logger.debug(f"App. magnitude is {app_mag:.2f}")
    return {"abs_mag": abs_mag, "app_mag": app_mag}


def get_flux_from_magnitude(mag):
    """
    Convert AB magnitude to flux

    Parameters
    ----------
    mag : float, array-like
        magnitudes to convert

    Returns
    -------
    : float, array-like
        flux values in Jy
    """
    const = 2.5 * np.log10(3631)
    return 10 ** ((mag - const) / -2.5)


class BasicInstrument(ABC):
    def __init__(self, fov, res):
        """
        Template class for defining basic observation instrument properties

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        res : float
            angular resolution in arcsec/pixel
        """
        self.field_of_view = fov
        self.angular_resolution = res
        self._ang_scale = None
        self.max_extent = 40.0

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
    def pixel_width(self):
        self._param_check()
        return self.angular_resolution * self._ang_scale

    @property
    def extent(self):
        self._param_check()
        return min(self._ang_scale * self.field_of_view, self.max_extent)

    @property
    def number_pixels(self):
        phys_extent = self.extent
        return int(phys_extent / self.pixel_width)


class MUSE_NFM(BasicInstrument):
    def __init__(self):
        """
        MUSE narrow field mode instrument. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
        """
        super().__init__(fov=7.42, res=0.025)


class MUSE_WFM(BasicInstrument):
    def __init__(self):
        """
        MUSE wide field mode instrument. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
        """
        super().__init__(fov=60, res=0.2)


class Euclid_NISP(BasicInstrument):
    def __init__(self):
        """ "
        "Euclid infrared bands. Parameters taken from:
        https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_s_instruments
        """
        super().__init__(fov=0.55 * 3600, res=0.3)


class Euclid_VIS(BasicInstrument):
    def __init__(self):
        """ "
        "Euclid visible bands. Parameters taken from:
        https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_s_instruments
        """
        super().__init__(fov=0.56 * 3600, res=0.101)
