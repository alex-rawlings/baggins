import numpy as np
import scipy.stats


__all__ = ["iqr", "smooth_bootstrap", "stat_interval", "uniform_sample_sphere", "vertical_RMSE"]


def iqr(x):
    """
    Return the interquartile range of an array.

    Parameters
    ----------
    x: numpy array

    Returns
    -------
    interquartile range
    """
    return np.nanquantile(x, 0.75, axis=-1) - np.nanquantile(x, 0.25, axis=-1)

def smooth_bootstrap(data, number_resamples=1e4, sigma=None, statistic=np.std, rng=None):
    """
    Perform a smooth bootstrap resampling to estimate a statistic

    Parameters
    ----------
    data: array of data values to bootstrap. Accepts a (m,n) array, in which 
          case each column is bootstrapped independently
    number_resamples: number of resamples to perform
    sigma: spread in the smoothing random variable. Default is SE/sqrt(m), where
           m is the number of rows and SE is the standard error of the sample
    statistic: function whose statistic is to be estimated
    rng: numpy random number generator. If not given, a new RNG is created

    Returns
    -------
    bootstrap_stat: (number_resamples, n) array of statistic estimate at each
                    iteration
    means: mean of each statistic estimate
    """
    number_resamples = int(number_resamples)
    if sigma is None:
        sigma = 2*np.std(data, axis=0) / np.sqrt(data.shape[0])
    bootstrap_stat = np.full((number_resamples, data.shape[-1]), np.nan)
    if rng is None:
        rng = np.random.default_rng()
    for i in range(number_resamples):
        print("Bootstrapping {:.2f}% complete           ".format(i/(number_resamples-1)*100), end="\r")
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
    x: array of observed x data
    y: array of observed y data
    xnew: the values we wish to compute for
    type: [conf pred] either a confidence or prediction interval
    conf_lev: the confidence level, where the value (1-conf_lev) is the 
              the integral area for the t-distribution
    
    Returns
    -------
    upper: the upper interval bound
    lower: the lower interval bound
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
    n: number of points
    rng: numpy random number generator object. If not given, a new RNG is 
         created
    
    Returns
    -------
    theta, phi: angular coordinates of points
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
    x: array of observed x data
    y: array of observed y data
    return_linregress: bool, return the slope and intercept from the linear
                       regeression model

    Returns
    -------
    root mean square error
    slope: linear regression gradient (only if return_linregress is True)
    intercept: linear regression intercept (only if return_linregress is True)
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