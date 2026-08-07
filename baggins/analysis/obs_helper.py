import os.path
import numpy as np
from copy import copy
from pygad import UnitQty
import unyt
from synthesizer import grid, instruments, GRID_DIR
from synthesizer.filters import Filter
from astropy import cosmology
from baggins.env_config import _cmlogger, synthesizer_data
from baggins.utils import get_files_in_dir

__all__ = [
    "set_luminosity",
    "get_synthesizer_grid",
    "get_spectrum_ssp",
    "get_filter_collection",
    "get_filter_lam_range",
    "get_surface_brightness",
    "get_flux_from_magnitude",
    "EUCLID_FILTER_CODES",
    "HST_FILTER_CODES",
    "JWST_MIRI_FILTER_CODES",
    "JWST_NIRCam_FILTER_CODES",
]


_logger = _cmlogger.getChild(__name__)

EUCLID_FILTER_CODES = ["Euclid/VIS.vis"]
HST_FILTER_CODES = [
    "HST/ACS_HRC.F435W",
    "HST/ACS_HRC.F555W",
    "HST/ACS_HRC.F606W",
]
JWST_MIRI_FILTER_CODES = [
    f"JWST/MIRI.F{x}"
    for x in [
        "560W",
        "770W",
        "1000W",
        "1065C",
        "1140C",
        "1130W",
        "1280W",
        "1500W",
        "1550C",
        "1800W",
        "2100W",
        "2300C",
        "2550W",
    ]
]
JWST_NIRCam_FILTER_CODES = [
    f"JWST/NIRCam.F{x}"
    for x in [
        "070W",
        "090W",
        "115W",
        "140M",
        "150W",
        "162M",
        "164N",
        "150W2",
        "182M",
        "187N",
        "200W",
        "210M",
        "212N",
        "250M",
        "277W",
        "300M",
        "323N",
        "322W2",
        "335M",
        "356W",
        "360M",
        "405N",
        "410M",
        "430M",
        "444W",
        "460M",
        "466N",
        "470N",
        "480M",
    ]
]


def set_luminosity(snap, sed, z=0):
    """
    Set the bolometric luminosity and magnitude for a gas-free snapshot in-place.

    Parameters
    ----------
    snap : pygad.Snapshot
        snapshot to set fields for
    sed : synthesizer.Sed
        spectral energy distribution object
    z : float, optional
        redshift of source (for cosmological dimming), by default 0
    age_units : str, optional
        units for age, by default "Gyr"
    """
    try:
        assert set(snap.stars.all_blocks()).isdisjoint({"lum", "metallicity", "age"})
    except AssertionError:
        _logger.exception(
            "Cannot set blocks 'lum', 'metallicity', or 'age' to a custom value if they already exist!",
            exc_info=True,
        )
        raise
    # sed object doesn't store unyt conversions, so manually obtain the
    # conversion from erg/s to Lsol
    _sed = copy(sed)
    _sed.lnu *= snap.stars["mass"][0].view(np.ndarray)
    erg_per_s_per_Lsun = unyt.Lsun.get_conversion_factor(
        _sed.bolometric_luminosity.units
    )[0]
    snap.stars["lum"] = UnitQty(
        np.full(
            len(snap.stars),
            _sed.bolometric_luminosity.value / erg_per_s_per_Lsun / (1 + z) ** 4,
        ),
        units="Lsol",
    )


def get_synthesizer_grid(grid_name=None, grid_dir=None, **kwargs):
    """
    Get a synthesizer grid.

    Parameters
    ----------
    grid_name : str, optional
        SSP grid to query, by default None"
    grid_dir : str, optional
        directory of grid_name, by default None

    Returns
    -------
    g : synthesizer.grid.Grid
        grid object
    """
    if grid_name is None:
        grid_name = "bpass-2.2.1-bin_chabrier03-0.1,100.0.hdf5"
    if grid_dir is None:
        # use the default location for grids from baggins config
        grid_dir = synthesizer_data
        if grid_dir is None:
            # use the default install location from synthesizer
            grid_dir = GRID_DIR
    try:
        assert os.path.isfile(os.path.join(grid_dir, grid_name))
    except AssertionError:
        valid_grids = get_files_in_dir(grid_dir, ext=".hdf5", name_only=True)
        _logger.exception(f"No grid called {grid_name}. Valid grids are {valid_grids}")
        raise
    _logger.info(f"Using data {os.path.join(grid_dir, grid_name)}")
    # create the grid
    kwargs.setdefault("ignore_lines", True)
    return grid.Grid(grid_name, grid_dir=grid_dir, **kwargs)


def get_spectrum_ssp(
    age,
    metallicity,
    grid_name=None,
    grid_dir=None,
):
    """
    Get the spectrum of a population, given some age and metallicity. Assumes that all stellar mass contributes equally (i.e. no dust attenuation).

    Parameters
    ----------
    age : float
        age of a stellar particle (in yr)
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
    g = get_synthesizer_grid(grid_name=grid_name, grid_dir=grid_dir)
    log10age = np.log10(age)
    # extract the spectrum at the target age / metallicity
    grid_point = g.get_grid_point(log10ages=log10age, metallicity=metallicity)
    sed = g.get_sed(grid_point, spectra_type="incident")
    return g, sed


def get_filter_collection(g, instr, new_lam_size=1000):
    """
    Convenience function to return all instrument transmission filters.

    Parameters
    ----------
    g : synthesizer.grid.Grid
        grid object to query wavelengths from
    instr : str
        instrument name
    new_lam_size : int, optional
        resample grid with this many wavelength bins, by default 1000

    Returns
    -------
    instr_filters : synthesizer.FilterCollection
        collection of instrument filters
    """
    instr_filter_map = dict(
        euclid=EUCLID_FILTER_CODES,
        hst=HST_FILTER_CODES,
        jwst_miri=JWST_MIRI_FILTER_CODES,
        jwst_nircam=JWST_NIRCam_FILTER_CODES,
    )
    instr_filters = instruments.FilterCollection(
        filter_codes=instr_filter_map[instr.lower()], new_lam=g.lam
    )
    instr_filters.resample_filters(lam_size=new_lam_size)
    return instr_filters


def get_filter_lam_range(filter_codes, pad_frac=0.05):
    """
    Get the native wavelength range spanned by a set of filters, without
    requiring a synthesizer grid. Used to tailor a grid's wavelength
    sampling to where a specific instrument's filters actually have
    transmission, before the grid itself is built.

    Parameters
    ----------
    filter_codes : list of str
        SVO filter codes (e.g. "Euclid/VIS.vis")
    pad_frac : float, optional
        fractional padding applied to the combined [min, max] range, by
        default 0.05

    Returns
    -------
    lam_min, lam_max : float
        wavelength bounds in Angstrom
    """
    lam_min = np.inf
    lam_max = -np.inf
    for code in filter_codes:
        f = Filter(code)
        lam_min = min(lam_min, f.original_lam.min().to("angstrom").value)
        lam_max = max(lam_max, f.original_lam.max().to("angstrom").value)
    pad = (lam_max - lam_min) * pad_frac
    return lam_min - pad, lam_max + pad


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
