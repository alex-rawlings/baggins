import numpy as np
import scipy.stats


__all__ = ['stat_interval', 'vertical_RMSE']


def stat_interval(x, y, xnew, type='conf', conf_lev=0.68):
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
    if type == 'conf':
        part_2 = tstat * np.std(y) * np.sqrt(1/n + (xnew-x_avg)**2/Sxx)
    elif type == 'pred':
        part_2 = tstat * np.std(y) * np.sqrt(1 + 1/n + (xnew-x_avg)**2/Sxx)
    else:
        raise ValueError("type must be conf or pred!")
    upper = part_1 + part_2
    lower = part_1 - part_2
    return upper, lower


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