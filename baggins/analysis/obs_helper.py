import os.path
import numpy as np
from copy import copy
import pygad
from synthesizer import grid, instruments
from astropy import units, cosmology
from baggins.env_config import _cmlogger, synthesizer_data
from baggins.mathematics import get_pixel_value_in_image, empirical_cdf

__all__ = ["set_luminosity", "signal_prominence", "get_spectrum_ssp", "get_euclid_filter_collection", "get_magnitudes"]


_logger = _cmlogger.getChild(__name__)


def set_luminosity(snap, metallicity, age, z=0, age_units="Gyr", **kwargs):
    """
    Set the luminosity and magnitude for a gas-free snapshot in-place.

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
        assert set(snap.stars.all_blocks()).isdisjoint({"metallicity", "age"})
    except AssertionError:
        _logger.exception("Cannot set blocks 'metallicity' and 'age' to a custom value if they already exist!", exc_info=True)
        raise
    if "band" not in kwargs or kwargs["band"] is None:
        kwargs["band"] = "bolometric"
    _logger.info(f"Determining luminosity in {kwargs['band']} band")
    if kwargs["band"] != "bolometric":
        lum_name = f"lum_{kwargs['band'].lower()}"
    else:
        lum_name = "lum"
    # add the metallicity and age blocks to the snapshot
    if isinstance(metallicity, (float, int)):
        metallicity = pygad.UnitQty(np.full(len(snap.stars), metallicity), pygad.physics.solar.Z(), subs=snap)
    snap.stars["metallicity"] = metallicity
    if isinstance(age, (int, float)):
        age = pygad.UnitArr(np.full(len(snap.stars), age), units=age_units, subs=snap)
    snap.stars["age"] = age
    # determine the (by default) bolometric luminosity
    # TODO calculate cosmological dimming here or elsewhere?
    snap.stars[lum_name] = pygad.get_luminosities(snap.stars, **kwargs) / (1+z)**4


def signal_prominence(x, y, im, npix=3):
    """
    Determine the prominence of a pixel in a map relative to the surrounding 
    pixels.

    Parameters
    ----------
    x : float
        x coordinate
    y : float
        y coordinate
    im : ax.AxesImage
        output from plt.imshow() - the image to search
    npix : int, optional
        number of pixels to search within, by default 3

    Returns
    -------
    : float
        quantile of target pixel relative to surrounds
    """
    # determine the signal of the pixel we're after
    val, row, col = get_pixel_value_in_image(x, y, im)
    # get all pixel values in some radius, excluding the central one
    rows = np.arange(row-npix, row+npix+1, 1)
    rows = rows[rows != row]
    cols = np.arange(col-npix, col+npix+1, 1)
    cols = cols[cols != col]
    surrounds = im.get_array()[rows, cols].flatten()
    return empirical_cdf(surrounds, val)


def get_spectrum_ssp(age, metallicity, grid_name="bpass-2.2.1-bin_chabrier03-0.1,100.0.hdf5", grid_dir=None):
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
    euclid_filters = instruments.FilterCollection(
        filter_codes = [f"Euclid/NISP.{b}" for b in ("Y", "J", "H")],
        new_lam=g.lam
    )
    euclid_filters.resample_filters(lam_size=new_lam_size)
    return euclid_filters


def get_magnitudes(sed, stellar_mass, filters_collection, filter_code, z):
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

    Returns
    -------
    : dict
        absolute and apparent magnitudes
    """
    # need to copy the sed so we don't affect original
    _sed = copy(sed)
    # "super-impose" spectra from all the stellar mass
    _sed.lnu *= stellar_mass
    _sed.get_fnu(cosmo=cosmology.Planck18, z=z)

    chosen_filter = filters_collection[filter_code]

    Lvo = 3631 * units.jansky * 4*np.pi * (10*units.parsec)**2
    Lvo = Lvo.to("erg/(Hz*s)")
    _logger.debug(f"Lvo is {Lvo}")

    abs_mag = -2.5 * np.log10(
        chosen_filter.apply_filter(
            _sed.lnu, nu=_sed.obsnu
        ) / Lvo.value)
    _logger.debug(f"Abs. magntiude is: {abs_mag:.2f}")

    app_mag = abs_mag + 5 * np.log10(
        cosmology.Planck18.luminosity_distance(z).to("pc") / (10 * units.parsec)
        ) - 2.5 * np.log10(1 + z)
    _logger.debug(f"App. magnitude is {app_mag:.2f}")
    return {"abs_mag": abs_mag, "app_mag":app_mag}
