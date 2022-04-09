import numpy as np
import scipy.stats


__all__ = ["iqr", "smooth_bootstrap", "stat_interval", "uniform_sample_sphere", "vertical_RMSE"]


def iqr(x):
    """
    Return the interquartile range of an array.

    Parameters
    ----------
    x : np.ndarray
        observations

    Returns
    -------
    : float
        interquartile range
    """
    return np.nanquantile(x, 0.75, axis=-1) - np.nanquantile(x, 0.25, axis=-1)


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
        sigma = 2*np.std(data, axis=0) / np.sqrt(data.shape[0])
    bootstrap_stat = np.full((number_resamples, data.shape[-1]), np.nan)
    if rng is None:
        rng = np.random.default_rng()
    for i in range(number_resamples):
        print(f"Bootstrapping {i/(number_resamples-1)*100:.2f}% complete           ", end="\r")
        #resample data columnwise
        resampled_data = rng.choice(data, data.shape[0], replace=True, axis=0)
        bootstrap_data = rng.normal(resampled_data, sigma)
        bootstrap_stat[i, :] = statistic(bootstrap_data, axis=0)
    print("Bootstrap complete                                ")
    return bootstrap_stat, np.mean(bootstrap_stat, axis=0)


def stat_interval(x, y, xnew, type="conf", conf_lev=0.68):
    """
    Determine either a confidence or prediction interval given some data. The
    approach follows that of "Mathematical Statistics (Wackerly) 7ed pg 595".

    Parameters
    ----------
    x : np.ndarray
        observed x data
    y : np.ndarray
        observed y data
    xnew : np.ndaray
        values we wish to compute for
    type : str, optional
        either a confidence or prediction interval, by default "conf"
    conf_lev : float, optional
        confidence level, where the value (1-conf_lev) is the the integral area 
        for the t-distribution, by default 0.68

    Returns
    -------
    upper : np.ndarray
        upper interval bound
    lower : np.ndarray
        lower interval bound

    Raises
    ------
    AssertionError
        conf_level must be in range (0,1)
    ValueError
        type must be one of [conf, pred]
    """
    assert(conf_lev<1 and conf_lev>0)
    #clean data
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    #fit a linear regression model
    slope, intercept, *_ = scipy.stats.linregress(x, y)
    #quantities we will need later
    x_avg = np.mean(x)
    Sxx = np.sum((x - x_avg)**2)
    n = len(x)
    #determine the t_{alpha/2} statistic
    tstat = scipy.stats.t.ppf((1-conf_lev)/2, n-2)
    #this is the part from the model
    part_1 = intercept + slope * xnew
    #and below this is the part from the error estimate
    if type == "conf":
        part_2 = tstat * np.std(y) * np.sqrt(1/n + (xnew-x_avg)**2/Sxx)
    elif type == "pred":
        part_2 = tstat * np.std(y) * np.sqrt(1 + 1/n + (xnew-x_avg)**2/Sxx)
    else:
        raise ValueError("type must be conf or pred!")
    upper = part_1 + part_2
    lower = part_1 - part_2
    return upper, lower


def uniform_sample_sphere(n, rng=None):
    """
    Uniformly sample points on the unit sphere.

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
