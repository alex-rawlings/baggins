import numpy as np
import scipy.stats
from ..env_config import _cmlogger


__all__ = ["iqr", "quantiles_relative_to_median", "smooth_bootstrap", "stat_interval", "uniform_sample_sphere", "vertical_RMSE"]

_logger = _cmlogger.copy(__file__)


def iqr(x, axis=-1):
    """
    Return the interquartile range of an array.

    Parameters
    ----------
    x : np.ndarray
        observations
    axis : int, optional
        axis to apply IQR over

    Returns
    -------
    : float
        interquartile range
    """
    return np.nanquantile(x, 0.75, axis=axis) - np.nanquantile(x, 0.25, axis=axis)


def quantiles_relative_to_median(x, lower=0.25, upper=0.75, axis=-1):
    """
    Determine difference of an upper and lower quantile from the median, useful
    for plotting quantile error bars with pyplot.

    Parameters
    ----------
    x : array-like
        array to determine quantiles of
    lower : float, optional
        lower quantile, by default 0.25
    upper : float, optional
        upper quantile, by default 0.75
    axis : int, optional
        axis to apply the operation over, by default -1

    Returns
    -------
    m : array-like
        array of median values
    spread : array-like
        array of shape (n,2) of lower, upper quantile pairs
    """
    m = np.nanmedian(x, axis=axis)
    try:
        assert lower < 0.5 and upper > 0.5
    except AssertionError:
        _logger.logger.exception(f"Lower quantile {lower} must be less than 0.5 Upper quantile {upper} must be greater than 0.5", exc_info=True)
    l = m - np.nanquantile(x, lower, axis=axis)
    u = np.nanquantile(x, upper, axis=axis) - m
    # convert to shape convenient for plotting with pyplot.errorbar
    spread = np.vstack((l,u)).T
    return m, spread


def smooth_bootstrap(data, number_resamples=1e4, sigma=None, statistic=np.std, rng=None):
    """
    Perform a smooth bootstrap resampling to estimate a statistic

    Parameters
    ----------
    data : np.ndarray
        data values to bootstrap. Accepts a (m,n) array, in which 
        case each column is bootstrapped independently
    number_resamples : int, float, optional
        number of resamples to perform, by default 1e4
    sigma : float, optional
        spread in the smoothing random variable, by default None (SE/sqrt(m), 
        where m is the number of rows and SE is the standard error of the 
        sample)
    statistic : function, optional
        np function statistic to be estimated, by default np.std
    rng : np.random._generator.Generator, optional
        random number generator, by default None (creates a new instance)

    Returns
    -------
    bootstrap_stat : np.ndarray
        array of statistic estimate at each iteration, shape (number_resamples, 
        n)
    : np.ndarray
        mean of each statistic estimate
    """
    number_resamples = int(number_resamples)
    if sigma is None:
        sigma = 2*np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
    bootstrap_stat = np.full((number_resamples, data.shape[-1]), np.nan)
    if rng is None:
        rng = np.random.default_rng()
    for i in range(number_resamples):
        print(f"Bootstrapping {i/(number_resamples-1)*100:.2f}% complete           ", end="\r")
        #resample data columnwise
        resampled_data = rng.choice(data, data.shape[0], replace=True, axis=0)
        bootstrap_data = rng.normal(resampled_data, sigma)
        bootstrap_stat[i, :] = statistic(bootstrap_data, axis=0)
    _logger.logger.info("Bootstrap complete                                ")
    return bootstrap_stat, np.nanmean(bootstrap_stat, axis=0)


def stat_interval(x, y, type="conf", conf_lev=0.68):
    """
    _summary_

    Parameters
    ----------
    x : array-like
        observed independent data
    y : array-like
        observed dependent data
    type : str, optional
        confidence interval for mean or prediction interval, by default "conf"
    conf_lev : float, optional
        confidence level, where the value (1-conf_lev) is the the integral area 
        for the t-distribution, by default 0.68

    Returns
    -------
    : callable
        function for error estimate
    """
    try:
        assert(conf_lev<1 and conf_lev>0)
    except AssertionError:
        _logger.logger.exception(f"Confidence level must be between 0 and 1, not {conf_lev}!", exc_info=True)
        raise
    try:
        assert type in ("conf", "pred")
    except AssertionError:
        _logger.logger.exception(f"Type {type} is not valid! Must be one of 'conf' or 'pred'!", exc_info=True)
        raise
    #clean data
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    #quantities we will need later
    x_avg = np.mean(x)
    Sxx = np.sum((x - x_avg)**2)
    n = len(x)
    #determine the t_{alpha/2} statistic
    tstat = scipy.stats.t.ppf((1-conf_lev)/2, n-2)
    #and below this is the part from the error estimate
    if type == "conf":
        return lambda u: tstat * np.std(y) * np.sqrt(1/n + (u-x_avg)**2/Sxx)
    else:
        return lambda u: tstat * np.std(y) * np.sqrt(1 + 1/n + (u-x_avg)**2/Sxx)


def uniform_sample_sphere(n, rng=None):
    """
    Uniformly sample points on the unit sphere assuming the physics standard, 
    i.e.:
        0 < theta < pi
        0 < phi < 2*pi

    Parameters
    ----------
    n : int
        number of points
    rng : np.random._generator.Generator, optional
        random number generator, by default None (creates a new instance)

    Returns
    -------
    theta : np.ndarray
        angular coordinates of points
    phi : np.ndarray
        angular coordinates of points
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = np.arccos(2*rng.uniform(size=n)-1)
    phi = 2 * np.pi * rng.uniform(size=n)
    return theta, phi


def vertical_RMSE(x, y, return_linregress=False):
    """
    Determine the root-mean square error of the data y coordinate

    Parameters
    ----------
    x : np.ndarray
        observed x data
    y : np.ndarray
        observed y data
    return_linregress : bool, optional
        return the slope and intercept from the linear regeression model?, by 
        default False

    Returns
    -------
    : np.ndarray
        root mean square error
    slope : float, optional
         linear regression gradient
    intercept : float, optional
        inear regression intercept
    """
    #clean data
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    slope, intercept, *_ = scipy.stats.linregress(x, y)
    yhat = slope*x+intercept
    if return_linregress:
        return np.sqrt(np.sum((yhat-y)**2)/len(x)), slope, intercept
    else:
        return np.sqrt(np.sum((yhat-y)**2)/len(x))
