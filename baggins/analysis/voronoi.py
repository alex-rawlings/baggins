from tqdm.dask import TqdmCallback
import numpy as np
import scipy.optimize
import scipy.ndimage
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from voronoi_binning import voronoi_binned_image
from baggins.env_config import _cmlogger
from baggins.mathematics import get_histogram_bin_centres
from pygad import UnitArr
import dask

__all__ = [
    "VoronoiKinematics",
    "lambda_R",
    "radial_profile_velocity_moment",
]

_logger = _cmlogger.getChild(__name__)


class VoronoiKinematics:
    def __init__(self, x, y, V, m, Npx=100, seeing=None):
        """
        Class to create voronoi kinematic maps

        Parameters
        ----------
        x : array-like
            particle x positions
        y : array-like
            particle y-positions
        V : array-like
            LOS velocities
        m : array-like
            particle masses
        Npx : int, optional
            number of pixels to create the voronoi from, by default 100
        seeing : dict, optional
            contaminate image, must contain keys 'num' and 'sigma', by default
            None
        """
        try:
            assert len(x) == len(y) == len(V) == len(m)
        except AssertionError:
            _logger.exception("Input arrays must be of the same length!", exc_info=True)
            raise

        if seeing is not None:
            try:
                for k in ("num", "sigma"):
                    assert k in seeing.keys()
            except AssertionError:
                _logger.exception(
                    f"Key {k} is not present in dict `seeing`!", exc_info=True
                )
                raise
            rng = seeing.setdefault("rng", np.random.default_rng())
            x = np.array(
                [xx + rng.normal(0, seeing["sigma"], size=seeing["num"]) for xx in x]
            ).flatten()
            y = np.array(
                [yy + rng.normal(0, seeing["sigma"], size=seeing["num"]) for yy in y]
            ).flatten()
            V = np.repeat(V, seeing["num"]).flatten()
            m = np.repeat(m / seeing["num"], seeing["num"]).flatten()
        else:
            _logger.warning("No seeing correction will be applied!")

        self.M = np.sum(m)
        xcom = np.sum(m * x) / self.M
        ycom = np.sum(m * y) / self.M
        Vcom = np.sum(m * V) / self.M
        self.m = m
        self.x = x - xcom
        self.y = y - ycom
        self.vz = V - Vcom
        self.Npx = Npx
        self.seeing = seeing
        self._extent = None
        self._grid = None
        self._stats = None

    @property
    def extent(self):
        return self._extent

    @property
    def max_voronoi_bin_index(self):
        return int(np.max(self._grid["particle_vor_bin_num"]) + 1)

    def _stat_is_calculated(self, p):
        try:
            assert self._stats is not None
        except AssertionError:
            _logger.exception(
                f"Need to determine LOS statistics first before calling property '{p}'!",
                exc_info=True,
            )
            raise

    @property
    def img_V(self):
        self._stat_is_calculated("V")
        return self._stats["img_V"]

    @property
    def img_sigma(self):
        self._stat_is_calculated("sigma")
        return self._stats["img_sigma"]

    @property
    def img_h3(self):
        self._stat_is_calculated("h3")
        return self._stats["img_h3"]

    @property
    def img_h4(self):
        self._stat_is_calculated("h4")
        return self._stats["img_h4"]

    @property
    def stats(self):
        return self._stats

    def make_grid(self, extent=None, part_per_bin=500):
        """
        Create the voronoi grid

        Parameters
        ----------
        extent : float, optional
            extent of image, by default None
        part_per_bin : int, optional
            target number of particles per bin, by default 500
        """
        if isinstance(extent, (float, int)):
            extent = (extent, extent)
        self._extent = extent
        # bin in the x-y plane
        nimg, xedges, yedges, grid_bin_num = binned_statistic_2d(
            self.x,
            self.y,
            values=None,
            statistic="count",
            bins=self.Npx,
            range=self.extent,
            expand_binnumbers=True,
        )

        # determine image extent
        w = xedges[-1] - xedges[0]
        h = yedges[-1] - yedges[0]
        assert w > 0 and h > 0

        # assign particles to voronoi bin
        pixel_vor_bin_num = voronoi_binned_image(
            nimg, part_per_bin, w, h, use_geometric_centroids_in_initial_binning=True
        )
        particle_vor_bin_num = np.full(self.x.shape, -1, dtype=int)
        valid_grid_bin_mask = np.logical_and(
            np.all(grid_bin_num > 0, axis=0),
            np.all(grid_bin_num < self.Npx + 1, axis=0),
        )
        indx, indy = grid_bin_num[:, valid_grid_bin_mask] - 1
        particle_vor_bin_num[valid_grid_bin_mask] = pixel_vor_bin_num[indy, indx]

        if self.extent is None:
            self._extent = (*xedges[[0, -1]], *yedges[[0, -1]])

        # create mesh
        X, Y = np.meshgrid(
            get_histogram_bin_centres(xedges), get_histogram_bin_centres(yedges)
        )
        index = np.unique(pixel_vor_bin_num)
        bin_sums = scipy.ndimage.sum(nimg, labels=pixel_vor_bin_num, index=index)
        x_bin = (
            scipy.ndimage.sum(nimg * X, labels=pixel_vor_bin_num, index=index)
            / bin_sums
        )
        y_bin = (
            scipy.ndimage.sum(nimg * Y, labels=pixel_vor_bin_num, index=index)
            / bin_sums
        )
        self._grid = dict(
            particle_vor_bin_num=particle_vor_bin_num,
            pixel_vor_bin_num=pixel_vor_bin_num,
            x_bin=x_bin,
            y_bin=y_bin,
            xedges=xedges,
            yedges=yedges,
        )

    def gauss_hermite_function(self, x, mu, sigma, h3, h4):
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

    @dask.delayed
    def fit_gauss_hermite_distribution(self, data):
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
            ll = -np.nansum(
                np.log(self.gauss_hermite_function(data, mu0, sigma0, *pars))
            )
            return ll

        try:
            res = scipy.optimize.least_squares(
                log_likelihood, (0.0, 0.0), loss="huber", bounds=((-1, -1), (1, 1))
            )
            h3, h4 = res.x
        except ValueError as err:
            _logger.warning(
                f"Unsuccessful fitting of Gauss-Hermite function for IFU bin - {err}"
            )
            h3, h4 = np.nan, np.nan
        return mu0, sigma0, h3, h4

    def binned_LOSV_statistics(self):
        """
        Generate the binned LOS statistics for the voronoi map
        """
        _logger.info(f"Binning {len(self.x):.2e} particles...")
        bin_index = list(range(self.max_voronoi_bin_index))

        bin_mass = scipy.ndimage.sum(
            self.m, labels=self._grid["particle_vor_bin_num"], index=bin_index
        )

        try:
            assert max(bin_index) > 1
        except AssertionError:
            _logger.exception("We only have one voronoi bin!", exc_info=True)
            raise
        fits = []
        for i in bin_index:
            fits.append(
                self.fit_gauss_hermite_distribution(
                    self.vz[self._grid["particle_vor_bin_num"] == i]
                )
            )
        with TqdmCallback(desc="Fitting voronoi bins"):
            fits = dask.compute(*fits)
        bin_stats = np.array(fits)
        img_stats = bin_stats[self._grid["pixel_vor_bin_num"]]

        self._stats = dict(
            bin_V=bin_stats[:, 0],
            bin_sigma=bin_stats[:, 1],
            bin_h3=bin_stats[:, 2],
            bin_h4=bin_stats[:, 3],
            bin_mass=bin_mass,
            img_V=img_stats[..., 0],
            img_sigma=img_stats[..., 1],
            img_h3=img_stats[..., 2],
            img_h4=img_stats[..., 3],
        )

    def binned_LOS_dispersion_only(self, ax=None, clims=None, cbar="adj"):
        """
        Fit only the LOSVD variance (no Gauss Hermite fitting) and plot.

        Parameters
        ----------
        ax : pyplot.Axes, optional
            plotting axis, by default None
        clims : tuple, optional
            colour limits for dispersion, by default None
        cbar : str, optional
            how to add a colourbar to each map, can be "adj" for adjacent or "inset" for inset, by default "adj"

        Returns
        -------
        ax : pyplot.Axes
            plotting axis
        """
        _logger.info(f"Binning {len(self.x):.2e} particles...")
        bin_index = list(range(self.max_voronoi_bin_index))
        try:
            assert max(bin_index) > 1
        except AssertionError:
            _logger.exception("We only have one voronoi bin!", exc_info=True)
            raise

        fits = []
        for i in bin_index:
            fits.append(
                dask.delayed(np.nanstd)(
                    self.vz[self._grid["particle_vor_bin_num"] == i]
                )
            )
        with TqdmCallback(desc="Fitting voronoi bins"):
            fits = dask.compute(*fits)
        bin_stats = np.array(fits)
        img_stats = bin_stats[self._grid["pixel_vor_bin_num"]]

        # now plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        cmap = colormaps.get("voronoi_seq")
        cmap.set_bad(color="k")
        if clims is None:
            norm = colors.Normalize(img_stats.min(), img_stats.max())
        else:
            norm = colors.Normalize(*clims)
        ax.set_aspect("equal")
        p1 = ax.imshow(
            img_stats,
            interpolation="nearest",
            origin="lower",
            extent=self.extent,
            cmap=cmap,
            norm=norm,
        )
        label = r"$\sigma/\mathrm{km}\,\mathrm{s}^{-1}$"
        if cbar == "adj":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(p1, cax=cax, label=label)
        elif cbar == "inset":
            cax = ax.inset_axes([0.4, 0.95, 0.55, 0.025])
            cax.set_xticks([])
            cax.set_yticks([])
            cax.patch.set_alpha(0)
            plt.colorbar(p1, cax=cax, label=label, orientation="horizontal")
        else:
            _logger.debug("No colour bar added")
        return ax

    def plot_kinematic_maps(
        self, ax=None, figsize=(7, 4.7), clims={}, desat=False, cbar="adj"
    ):
        """
        Plot the voronoi maps for a system.

        Parameters
        ----------
        vdat : dict
            voronoi values from baggins.analysis.voronoi_binned_los_V_statistics()
        ax : np.ndarray, optional
            numpy array of pyplot.Axes objects for plotting, by default None
        figsize : tuple, optional
            figure size, by default (7,4.7)
        clims : dict, optional
            colour scale limits, by default None
        desat : bool, optional
            use a desaturated colour scheme, by default False
        cbar : str, optional
            how to add a colourbar to each map, can be "adj" for adjacent or "inset" for inset, by default "adj"

        Returns
        -------
        ax : np.ndarray
            plotting axes
        """
        self._stat_is_calculated("V")
        # set the colour limits
        _clims = dict(V=[None], sigma=[None, None], h3=[None], h4=[None])
        for k, v in clims.items():
            try:
                assert isinstance(v, (list, tuple))
            except AssertionError:
                _logger.exception(
                    f"Each value of `clim` must be a list or tuple, not {type(v)}!",
                    exc_info=True,
                )
                raise
            vlen = 2 if k == "sigma" else 1
            try:
                assert len(v) == vlen
            except AssertionError:
                _logger.exception(
                    f"`clim` entry for {k} must be of length {vlen}, not {len(v)}!",
                    exc_info=True,
                )
                raise
            _clims[k] = v

        # set up the figure
        if ax is None:
            fig, ax = plt.subplots(2, 2, sharex="all", sharey="all", figsize=figsize)
            for i in range(2):
                ax[1, i].set_xlabel(r"$x/\mathrm{kpc}$")
                ax[i, 0].set_ylabel(r"$y/\mathrm{kpc}$")
        if desat:
            div_cols = colormaps.get("voronoi_div_desat")
            asc_cols = colormaps.get("voronoi_seq_desat")
        else:
            div_cols = colormaps.get("voronoi_div")
            asc_cols = colormaps.get("voronoi_seq")
        for i, (statkey, axi, cmap, label) in enumerate(
            zip(
                ("V", "sigma", "h3", "h4"),
                ax.flat,
                (div_cols, asc_cols, div_cols, div_cols),
                (
                    r"$V/\mathrm{km}\,\mathrm{s}^{-1}$",
                    r"$\sigma/\mathrm{km}\,\mathrm{s}^{-1}$",
                    r"$h_3$",
                    r"$h_4$",
                ),
            )
        ):
            # plot the statistic
            cmap.set_bad(color="k")
            stat = self._stats[f"img_{statkey}"]
            if i != 1:
                norm = colors.CenteredNorm(vcenter=0, halfrange=_clims[statkey][0])
            else:
                if _clims["sigma"]:
                    norm = colors.Normalize(*_clims["sigma"])
                else:
                    norm = colors.Normalize(stat.min(), stat.max())
            axi.set_aspect("equal")
            p1 = axi.imshow(
                stat,
                interpolation="nearest",
                origin="lower",
                extent=self.extent,
                cmap=cmap,
                norm=norm,
            )
            if cbar == "adj":
                divider = make_axes_locatable(axi)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(p1, cax=cax, label=label)
            elif cbar == "inset":
                cax = axi.inset_axes([0.4, 0.95, 0.55, 0.025])
                cax.set_xticks([])
                cax.set_yticks([])
                cax.patch.set_alpha(0)
                plt.colorbar(p1, cax=cax, label=label, orientation="horizontal")
            else:
                _logger.debug("No colour bar added")
        return ax

    def get_pixel_LOSVD(self, x, y):
        """
        Return the LOSVD of a Voronoi pixel at a given coordinate.

        Parameters
        ----------
        x : float
            desired x-coordinate
        y : float
            desired y-coordinate

        Returns
        -------
        : array-like
            LOS velocity values of Voronoi pixel
        indx : int
            which regular pixel the coordinate is at (row)
        indy : int
            which regular pixel the coordinate is at (column)
        """
        # get the pixel number of x and y
        try:
            assert isinstance(x, (float, int)) and isinstance(y, (float, int))
        except AssertionError:
            _logger.exception(
                f"Coordinates must be a single point, not {type(x)}", exc_info=True
            )
            raise
        indx = np.digitize(x, self._grid["xedges"]) - 1
        indy = np.digitize(y, self._grid["yedges"]) - 1
        vor_bin_idx = self._grid["pixel_vor_bin_num"][indx, indy]
        return self.vz[self._grid["particle_vor_bin_num"] == vor_bin_idx], indx, indy

    def plot_pixel_LOSVD(self, x, y, ax=None, **kwargs):
        """
        Plot the LOSVD for a specific position

        Parameters
        ----------
        x : float
            desired x-coordinate
        y : float
            desired y-coordinate
        ax : pyplot.Axes, optional
            plotting axes, by default None

        Returns
        -------
        ax : pyplot.Axes, optional
            plotting axes, by default None
        """
        sample, indx, indy = self.get_pixel_LOSVD(x, y)
        # histogram particle velocities
        if ax is None:
            fig, ax = plt.subplots()
        h = ax.hist(sample, density=True, **kwargs)

        # add the LOSD over the top
        _v = np.linspace(h[1][0], h[1][-1], 1000)
        ax.plot(
            _v,
            self.gauss_hermite_function(
                _v,
                self.img_V[indx, indy],
                self.img_sigma[indx, indy],
                self.img_h3[indx, indy],
                self.img_h4[indx, indy],
            ),
        )
        return ax

    def lambda_R_parameter(self):
        R = np.sqrt(self._grid["x_bin"] ** 2 + self._grid["y_bin"] ** 2)
        idx = np.argsort(R)
        R = R[idx]
        F = self._stats["bin_mass"][idx]
        V = self._stats["bin_V"][idx]
        sig = self._stats["bin_sigma"][idx]
        lam = np.nancumsum(F * R * np.abs(V)) / np.nancumsum(
            F * R * np.sqrt(V**2 + sig**2)
        )
        return lambda x: np.interp(x, R, lam)


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


def lambda_R(vorstat):
    """
    Determine the lambda(R) spin parameter.
    Original form by Matias Mannerkoski

    Parameters
    ----------
    vorstat : dict
        output of voronoi_binned_los_V_statistics() method

    Returns
    -------
    : callable
        interpolation function to get spin value at a particular radius
    """
    R, inds = _get_R(vorstat)
    F = vorstat["bin_mass"][inds]
    V = vorstat["bin_V"][inds]
    s = vorstat["bin_sigma"][inds]
    lam = np.nancumsum(F * R * np.abs(V)) / np.nancumsum(F * R * np.sqrt(V**2 + s**2))
    return lambda x: np.interp(x, R, lam)


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
    return R, vorstat[f"bin_{stat}"][inds]
