import numpy as np
import scipy.optimize
import scipy.ndimage
from scipy.stats import binned_statistic_2d
from voronoi_binning import voronoi_binned_image
from ..env_config import _cmlogger
from pygad import UnitArr

__all__ = [
    "voronoi_grid",
    "gauss_hermite_function",
    "fit_gauss_hermite_distribution",
    "voronoi_binned_los_V_statistics",
    "lambda_R",
    "radial_profile_velocity_moment",
]

_logger = _cmlogger.getChild(__name__)


def voronoi_grid(x, y, Npx=100, extent=None, part_per_bin=500):
    """
    Create the voronoi grid by assigning particles to correct bins.
    Original form by Matias Mannerkoski

    Parameters
    ----------
    x : np.ndarray
        x coordinates of particles
    y : np.ndarray
        y coordinates of particles
    Npx : int, optional
        number of bins, by default 100
    extent : float, optional
        image extent, by default None (allows for full image to be plotted)
    part_per_bin : int, optional
        target number of particles per bin, by default 500

    Returns
    -------
    particle_vor_bin_num : np.ndarray
        voronoi bin number of particles
    pixel_vor_bin_num : np.ndarray
        voronoi bin number of pixels
    extent : tuple
        range to bin dimensions over
    x_bin : np.ndarray
        x coordinate of bin centres
    y_bin : np.ndarray
        y coordinate of bin centres
    """
    # bin in the x-y plane
    nimg, xedges, yedges, grid_bin_num = binned_statistic_2d(
        x,
        y,
        values=None,
        statistic="count",
        bins=Npx,
        range=extent,
        expand_binnumbers=True,
    )

    # determine image extent
    w = xedges[-1] - xedges[0]
    h = yedges[-1] - yedges[0]

    # assign particles to voronoi bin
    pixel_vor_bin_num = voronoi_binned_image(nimg, part_per_bin, w, h)
    particle_vor_bin_num = np.full(x.shape, -1, dtype=int)
    valid_grid_bin_mask = np.logical_and(
        np.all(grid_bin_num > 0, axis=0), np.all(grid_bin_num < Npx + 1, axis=0)
    )
    indx, indy = grid_bin_num[:, valid_grid_bin_mask] - 1
    particle_vor_bin_num[valid_grid_bin_mask] = pixel_vor_bin_num[indy, indx]

    if extent is None:
        extent = (*xedges[[0, -1]], *yedges[[0, -1]])

    # create mesh
    X, Y = np.meshgrid((xedges[1:] + xedges[-1:]) / 2, (yedges[1:] + yedges[-1:]) / 2)
    index = np.unique(pixel_vor_bin_num)
    bin_sums = scipy.ndimage.sum(nimg, labels=pixel_vor_bin_num, index=index)
    x_bin = (
        scipy.ndimage.sum(nimg * X, labels=pixel_vor_bin_num, index=index) / bin_sums
    )
    y_bin = (
        scipy.ndimage.sum(nimg * Y, labels=pixel_vor_bin_num, index=index) / bin_sums
    )

    return particle_vor_bin_num, pixel_vor_bin_num, extent, x_bin, y_bin


def gauss_hermite_function(x, mu, sigma, h3, h4):
    """
    The normalised (when non-negative) function corresponding to the first
    three terms in the expansion used by van der Marel & Franx
    (1993ApJ...407..525V) and others following their methods.
    Original form by Matias Mannerkoski.

    Parameters
    ----------
    x : np.ndarray
        data values
    mu : float
        mean of x
    sigma : float
        standard deviation of x
    h3 : float
        3rd moment
    h4 : float
        4th moment

    Returns
    -------
    : np.ndarray
        function value of x
    """
    w = (x - mu) / sigma
    a = np.exp(-0.5 * w**2) / np.sqrt(2 * np.pi)
    H3 = (2 * w**3 - 3 * w) / np.sqrt(3)
    H4 = (4 * w**4 - 12 * w**2 + 3) / np.sqrt(24)
    N = (
        np.sqrt(6) * h4 * sigma / 4 + sigma
    )  # normalization when the function is non-negative
    # TODO for fitting the function should be always normalized
    return np.clip(a * (1 + h3 * H3 + h4 * H4) / N, 1e-30, None)


def fit_gauss_hermite_distribution(data):
    """
    Fit a Gauss-Hermite distribution to the data.
    Original form by Matias Mannerkoski.

    Parameters
    ----------
    data : np.ndarray
         data to fit function for

    Returns
    -------
    mu0 : float
        fit mean
    sigma0 : float
        fit standard deviation
    h3 : float
        3rd moment
    h4 : float
        4th moment
    """
    if isinstance(data, UnitArr):
        data = data.view(np.ndarray)
    if len(data) == 0:
        _logger.warning("Data has length zero!")
        return 0.0, 0.0, 0.0, 0.0
    if np.any(np.isnan(data)):
        _logger.warning("Removing NaNs from data!")
        data = data[~np.isnan(data)]
    # the gauss hermite function is made to have the same mean and sigma as the
    # plain gaussian so compute them with faster estimates
    mu0 = np.nanmean(data)
    sigma0 = np.nanstd(data)

    def log_likelihood(pars):
        ll = -np.nansum(np.log(gauss_hermite_function(data, mu0, sigma0, *pars)))
        return ll

    try:
        res = scipy.optimize.least_squares(log_likelihood, (0.0, 0.0), loss="huber", bounds=((-1, -1), (1, 1)))
        h3, h4 = res.x
    except ValueError as err:
        _logger.warning(
            f"Unsuccessful fitting of Gauss-Hermite function for IFU bin - {err}"
        )
        h3, h4 = np.nan, np.nan
    return mu0, sigma0, h3, h4


def voronoi_binned_los_V_statistics(x, y, V, m, Npx=100, seeing={}, **kwargs):
    """
    Determine the statistics of each voronoi bin.

    Parameters
    ----------
    x : np.ndarray
        x coordinates of bins
    y : np.ndarray
        y coordinates of bins
    V : np.ndarray
        LOS velocity
    m : np.ndarray
        masses
    Npx : int, optional
        number of pixels per voronoi bin, by default 100

    Returns
    -------
    : dict
        binned quantitites convereted to CoM frame
    """
    try:
        assert len(x) == len(y) == len(V) == len(m)
    except AssertionError:
        _logger.exception("Input arrays must be of the same length!", exc_info=True)
        raise
    M = np.sum(m)
    xcom = np.sum(m * x) / M
    ycom = np.sum(m * y) / M
    Vcom = np.sum(m * V) / M
    x = x - xcom
    y = y - ycom
    vz = V - Vcom

    if seeing:
        try:
            for k in ("num", "sigma"):
                assert k in seeing.keys()
        except AssertionError:
            _logger.exception(
                f"Key {k} is not present in dict `seeing`!", exc_info=True
            )
            raise
        rng = seeing["rng"] if "rng" in seeing else np.random.default_rng()
        x = np.array(
            [xx + rng.normal(0, seeing["sigma"], size=seeing["num"]) for xx in x]
        ).flatten()
        y = np.array(
            [yy + rng.normal(0, seeing["sigma"], size=seeing["num"]) for yy in y]
        ).flatten()
        vz = np.repeat(vz, seeing["num"]).flatten()
        m = np.repeat(m, seeing["num"]).flatten()
    else:
        _logger.warning("No seeing correction will be applied!")

    _logger.info(f"Binning {len(x)} particles...")
    particle_vor_bin_num, pixel_vor_bin_num, extent, xBar, yBar = voronoi_grid(
        x, y, Npx=Npx, **kwargs
    )
    bin_index = list(range(int(np.max(particle_vor_bin_num) + 1)))

    bin_mass = scipy.ndimage.sum(m, labels=particle_vor_bin_num, index=bin_index)

    fits = []
    for i in bin_index:
        print("Fitting bin:", i, end="\r")
        fits.append(fit_gauss_hermite_distribution(vz[particle_vor_bin_num == i]))
    bin_stats = np.array(fits)
    img_stats = bin_stats[pixel_vor_bin_num]

    return dict(
        xBar=xBar,
        yBar=yBar,
        bin_V=bin_stats[:, 0],
        bin_sigma=bin_stats[:, 1],
        bin_h3=bin_stats[:, 2],
        bin_h4=bin_stats[:, 3],
        bin_mass=bin_mass,
        img_V=img_stats[..., 0],
        img_sigma=img_stats[..., 1],
        img_h3=img_stats[..., 2],
        img_h4=img_stats[..., 3],
        extent=extent,
        xcom=xcom,
        ycom=ycom,
        Vcom=Vcom,
    )


def _get_R(vs):
    """
    Helper function to get radial value of voronoi bins

    Parameters
    ----------
    vs : dict
        output of voronoi_binned_los_V_statistics() method

    Returns
    -------
    R : np.ndarray
        radial values
    inds : np.ndarray
        sorted indices of radius
    """
    R = np.sqrt(vs["xBar"] ** 2 + vs["yBar"] ** 2)
    inds = np.argsort(R)
    return R[inds], inds


def lambda_R(vorstat, re):
    """
    Determine the lambda(R) spin parameter.
    Original form by Matias Mannerkoski

    Parameters
    ----------
    vorstat : dict
        output of voronoi_binned_los_V_statistics() method
    re : float
        projected half mass (or half light) radius

    Returns
    -------
    : np.ndarray
        radial values in units of Re
    : np.ndarray
        lambda(R) value
    """
    R, inds = _get_R(vorstat)
    F = vorstat["bin_mass"][inds]
    V = vorstat["bin_V"][inds]
    s = vorstat["bin_sigma"][inds]
    return R / re, np.nancumsum(F * R * np.abs(V)) / np.nancumsum(
        F * R * np.sqrt(V**2 + s**2)
    )


def radial_profile_velocity_moment(vorstat, stat):
    """
    Obtain a radial mass-weighted velocity moment profile from a Voronoi map.

    Parameters
    ----------
    vorstat : dict
        output of voronoi_binned_los_V_statistics() method
    stat : str
        velocity moment, one of: "V", "sigma", "h3", or "h4"

    Returns
    -------
    R : np.ndarray
        radial values
    : np.ndarray
        radial moment profile
    """
    R, inds = _get_R(vorstat)
    F = vorstat["bin_mass"][inds]
    return R, vorstat[f"bin_{stat}"][inds]
