import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from tqdm.dask import TqdmCallback
from tqdm import trange
import dask
from baggins.env_config import _cmlogger


__all__ = [
    "iqr",
    "quantiles_relative_to_median",
    "smooth_bootstrap",
    "smooth_bootstrap_pval",
    "permutation_sample_test",
    "stat_interval",
    "uniform_sample_sphere",
    "vertical_RMSE",
    "empirical_cdf",
    "EmpiricalCDF",
]

_logger = _cmlogger.getChild(__name__)


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
        _logger.exception(
            f"Lower quantile {lower} must be less than 0.5 Upper quantile {upper} must be greater than 0.5",
            exc_info=True,
        )
    lo = m - np.nanquantile(x, lower, axis=axis)
    up = np.nanquantile(x, upper, axis=axis) - m
    # convert to shape convenient for plotting with pyplot.errorbar
    spread = np.vstack((lo, up))
    return m, spread


def _smooth_bootstrap_sigma(data):
    return 2 * np.nanstd(data, axis=0) / np.sqrt(data.shape[0])


def smooth_bootstrap(
    data, number_resamples=1e4, sigma=None, statistic=np.std, rng=None
):
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
        np function statistic to be estimated (must accept an axis argument),
        by default np.std
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
        sigma = _smooth_bootstrap_sigma(data)
    bootstrap_stat = np.full((number_resamples, data.shape[-1]), np.nan)
    if rng is None:
        rng = np.random.default_rng()
    for i in trange(number_resamples, desc="Bootstrapping"):
        # resample data columnwise
        resampled_data = rng.choice(data, data.shape[0], replace=True, axis=0)
        bootstrap_data = rng.normal(resampled_data, sigma)
        bootstrap_stat[i, :] = statistic(bootstrap_data, axis=0)
    return bootstrap_stat, np.nanmean(bootstrap_stat, axis=0)


def smooth_bootstrap_pval(data, alpha=0.01, statistic=np.std, **kwargs):
    """
    Determine the p-value of some statistic of an observed sample compared to
    a smooth bootstrap sample.

    Parameters
    ----------
    data : array-like
        sample (can be multiple samples, sample sets must belong to the same
        column)
    alpha : float, optional
        significance level, by default 0.01
    statistic : function, optional
        statistic to test, must accept the 'axis' argument, by default np.std

    Returns
    -------
    pval : array-like
        p-value of the sample statistic
    decision : list
        which hypothesis to accept given the significance level
    """
    bootstrap_stat, bootstrap_stat_mean = smooth_bootstrap(
        data, statistic=statistic, **kwargs
    )
    # define empirical cdf
    data_stat = statistic(data, axis=0)
    sqrt_n = np.sqrt(data.shape[0])
    ecdf = (
        lambda t: 1
        / bootstrap_stat.shape[0]
        * np.sum(sqrt_n * (bootstrap_stat - data_stat) < t, axis=0)
    )
    pval = np.full(data.shape[1], np.nan)
    decision = ["" for _ in range(len(pval))]
    ecdf_val = ecdf(alpha)
    # determine equal tail p-value
    for i in range(len(pval)):
        pval[i] = 2 * min(ecdf_val[i], 1 - ecdf_val[i])
        decision[i] = "Ha" if pval[i] < alpha else "H0"
    return pval, decision


def permutation_sample_test(data1, data2, number_resamples=1e4, rng=None):
    """
    Perform a two-sample permutation test to determine if the variance between
    two samples, and thus if the distributions from which the samples are
    drawn, are different. The algorithm is 15.1 from "An Introduction to the
    Bootstrap" by Efron & Tibshirani

    Parameters
    ----------
    data1 : array-like
        sample set 1
    data2 : array-like
        sample set 2
    number_resamples : int, float, optional
        number of bootstrap samples, by default 1e4
    rng : np.random._generator.Generator, optional
        random number generator, by default None (creates a new instance)

    Returns
    -------
    : float
        achieved significance level of the test
    """
    try:
        assert len(data1.shape) == len(data2.shape) == 1
    except AssertionError:
        _logger.exception("Data must be 1 dimensional!", exc_info=True)
    number_resamples = int(number_resamples)
    if rng is None:
        rng = np.random.default_rng()
    # clean data of NaNs
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    n = len(data1)
    m = len(data2)
    data = np.concatenate((data1, data2))
    # define the test statistic
    tstatfun = lambda g: np.log10(np.nanvar(data[g]) / np.nanvar(data[~g]))
    group = np.full(n + m, 1, dtype=bool)
    group[n:] = 0
    tstat = tstatfun(group)
    data.sort()
    bootstrap_stat = np.full(number_resamples, np.nan)

    for i in range(number_resamples):
        print(
            f"Shuffling {i/(number_resamples-1)*100:.1f}% complete           ", end="\r"
        )
        shuffled_groups = rng.choice(group, group.shape[0], replace=False)
        bootstrap_stat[i] = tstatfun(shuffled_groups)
    _logger.info("Permutations complete                                ")
    return 2 * min(
        np.nanmean(bootstrap_stat < tstat), np.nanmean(bootstrap_stat > tstat)
    )


def stat_interval(x, y, itype="conf", conf_lev=0.68):
    """
    Determine the confidence or predictive interval of regression data

    Parameters
    ----------
    x : array-like
        observed independent data
    y : array-like
        observed dependent data
    itype : str, optional
        confidence interval for mean or prediction interval, by default "conf"
    conf_lev : float, optional
        confidence level, corresponding to the area 1-alpha of the
        t-distribution, (thus alpha is 1-conf_lev), by default 0.68

    Returns
    -------
    : callable
        function for error estimate
    """
    try:
        assert conf_lev < 1 and conf_lev > 0
    except AssertionError:
        _logger.exception(
            f"Confidence level must be between 0 and 1, not {conf_lev}!", exc_info=True
        )
        raise
    try:
        assert itype in ("conf", "pred")
    except AssertionError:
        _logger.exception(
            f"Type {itype} is not valid! Must be one of 'conf' or 'pred'!",
            exc_info=True,
        )
        raise
    # clean data
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    # quantities we will need later
    x_avg = np.mean(x)
    Sxx = np.sum((x - x_avg) ** 2)
    n = len(x)
    # determine the t_{alpha/2} statistic
    tstat = scipy.stats.t.ppf((1 - conf_lev) / 2, n - 2)
    # and below this is the part from the error estimate
    if itype == "conf":
        return lambda u: tstat * np.std(y) * np.sqrt(1 / n + (u - x_avg) ** 2 / Sxx)
    else:
        return lambda u: tstat * np.std(y) * np.sqrt(1 + 1 / n + (u - x_avg) ** 2 / Sxx)


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
    theta = np.arccos(2 * rng.uniform(size=n) - 1)
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
    # clean data
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    slope, intercept, *_ = scipy.stats.linregress(x, y)
    yhat = slope * x + intercept
    if return_linregress:
        return np.sqrt(np.sum((yhat - y) ** 2) / len(x)), slope, intercept
    else:
        return np.sqrt(np.sum((yhat - y) ** 2) / len(x))


def empirical_cdf(x, t):
    """
    Determine the empirical cumulative distribution function of an array

    Parameters
    ----------
    x : array-like
        observed data
    t : float
        value to determine cdf of

    Returns
    -------
    : float
        ECDF value at point
    """
    DeprecationWarning(
        "Empirical CDFs should be constructed using the new 'EmpiricalCDF' class"
    )
    return np.nanmean(x <= t)


class EmpiricalCDF:
    def __init__(self, x, weights=None):
        """
        Class to determine an empirical cumulative distribution function, that
        can also handle weights applied to the observations.

        Parameters
        ----------
        x : array-like
            observed data
        weights : array-like, optional
            weights, by default None
        """
        x = np.asarray(x)
        if weights is None:
            weights = np.ones_like(x, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)

        nan_idx = np.zeros_like(x, dtype=bool)
        nan_idx[np.isnan(x)] = True
        nan_idx[np.isnan(weights)] = True
        if np.sum(nan_idx) > 0:
            _logger.warning("NaNs detected, will be removed!")
            x = x[~nan_idx]
            weights = weights[~nan_idx]

        # sort data for consistent ECDF logic
        sort_idx = np.argsort(x)
        self.x_raw = x
        self.weights_raw = weights

        self.x_sorted = x[sort_idx]
        self.weights_sorted = weights[sort_idx]

        # combine duplicates: group by unique x and sum weights
        x_unique, inverse = np.unique(self.x_sorted, return_inverse=True)
        weight_sums = np.zeros_like(x_unique, dtype=float)
        np.add.at(weight_sums, inverse, self.weights_sorted)

        self.x = x_unique
        self.cdf_vals = np.cumsum(weight_sums)
        self.cdf_vals /= self.cdf_vals[-1]

    def cdf(self, x_val):
        """
        Evaluate the CDF.

        Parameters
        ----------
        x_val : array-like
            points to evaluate CDF at

        Returns
        -------
        : array-like
            CDF values
        """
        x_val = np.atleast_1d(x_val)
        return np.interp(x_val, self.x, self.cdf_vals, left=0.0, right=1.0)

    def ppf(self, q):
        """
        Evaluate the inverse CDF.

        Parameters
        ----------
        q : array-like
            quantiles to evaluate

        Returns
        -------
        : array-like
            distribution-sampeld value
        """
        q = np.atleast_1d(q)
        try:
            assert not np.any((q < 0) | (q > 1))
        except AssertionError:
            _logger.exception("Quantiles must be in [0, 1]", exc_info=True)
            raise
        idx = np.searchsorted(self.cdf_vals, q, side="right")
        idx = np.clip(idx, 0, len(self.x) - 1)
        return self.x[idx]

    def sample(self, size=1, random_state=None):
        """
        Sample the ECDF.

        Parameters
        ----------
        size : int, optional
            number of draws, by default 1
        random_state : float, optional
            random seed, by default None

        Returns
        -------
        : array-like
            sampled distribution values
        """
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0, 1, size)
        return self.ppf(u)

    def bootstrap_dirichlet(self, n_bootstraps=1000, random_state=None):
        """
        Bootstraps the data using a Dirichlet distribution for the weights.
        Preserves the weighted distribution using importance weights.

        Parameters
        ----------
        n_bootstraps : int, optional
            number of bootstraps, by default 1000
        random_state : float, optional
            random seed, by default None

        Returns
        -------
        boot_ecdfs : list
            list of bootstrapped ECDFs
        """
        rng = np.random.default_rng(random_state)
        boot_ecdfs = []
        alpha = self.weights_raw / self.weights_raw.sum() * len(self.weights_raw)

        @dask.delayed
        def _dask_helper():
            # use Dirichlet distribution to sample new weights
            new_weights = rng.dirichlet(alpha)
            return EmpiricalCDF(self.x_raw, new_weights)

        for _ in trange(n_bootstraps, desc="Initialising bootstrap"):
            boot_ecdfs.append(_dask_helper())
        with TqdmCallback(desc="Bootstrapping ECDF"):
            boot_ecdfs = dask.compute(*boot_ecdfs)
        return boot_ecdfs

    def plot(
        self,
        ax=None,
        ci_prob=None,
        npoints=100,
        n_bootstraps=500,
        ci_kwargs={},
        **kwargs,
    ):
        """
        Plot the observed ECDF, and optionally a bootstrapped-confidence
        interval if 'ci_prob' is not None.

        Parameters
        ----------
        ax : pyplot.Axes, optional
            plotting axes, by default None
        ci_prob : float, optional
            confidence interval, by default None
        npoints : int, optional
            number of points to evalute ECDF at, by default 100
        n_bootstraps : int, optional
            number of bootstrap draws, by default 500
        ci_kwargs : dict, optional
            plotting parameters given to pyplot.fill_between(), by default {}
        **kwargs :
            other keyword arguments given to pyplot.plot()

        Returns
        -------
        ax : pyplot.Axes, optional
            plotting axes, by default None
        """
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(np.nanmin(self.x), np.nanmax(self.x), npoints)
        (lp,) = ax.plot(x, self.cdf(x), **kwargs)
        if ci_prob is not None:
            try:
                assert 0 < ci_prob < 1
            except AssertionError:
                _logger.exception(
                    "Confidence interval must be between 0 and 1", exc_info=True
                )
                raise
            boots = self.bootstrap_dirichlet(n_bootstraps=n_bootstraps)
            cdf_vals_bootstrap = np.array([boot.cdf(x) for boot in boots])
            ax.fill_between(
                x,
                np.nanquantile(cdf_vals_bootstrap, 0.5 - ci_prob / 2, axis=0),
                np.nanquantile(cdf_vals_bootstrap, 0.5 + ci_prob / 2, axis=0),
                fc=lp.get_color(),
                zorder=lp.get_zorder() - 0.1,
                **ci_kwargs,
            )
        return ax
