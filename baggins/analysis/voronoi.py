from tqdm.dask import TqdmCallback
from tqdm import tqdm
import numpy as np
import scipy.optimize
import scipy.ndimage
import scipy.special
import scipy.integrate
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

__all__ = ["VoronoiKinematics", "unify_IFU_colour_scheme"]

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
        x = np.asarray(x)
        y = np.asarray(y)
        V = np.asarray(V)
        m = np.asarray(m)
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
        elif len(V) != 0:
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
        self._hermite_polys = []
        self._hermite_order = None

    @property
    def extent(self):
        return self._extent

    @property
    def max_voronoi_bin_index(self):
        return int(np.max(self._grid["particle_vor_bin_num"]) + 1)

    def _stat_is_calculated(self, k):
        try:
            assert self._stats is not None
        except AssertionError:
            _logger.exception(
                f"Need to determine LOS statistics first before calling property '{k}'!",
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
    def stats(self):
        return self._stats

    def get_colour_limits(self):
        """
        Get the colour limits of all kinematic maps for this object

        Returns
        -------
        d : dict
            colour limits to be parsed to plot_kinematic_maps()
        """
        d = {}
        d["V"] = np.max(np.abs(self.img_V))
        d["sigma"] = [np.min(self.stats["img_sigma"]), np.max(self.stats["img_sigma"])]
        for p in range(3, self._hermite_order + 1):
            d[f"h{p}"] = np.max(np.abs(self.stats[f"img_h{p}"]))
        return d

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

    def gauss_hermite_function(self, x, mu, sigma, *h):
        """
        Normalised Gauss-Hermite function (using physicist's definition)
        following van der Marel & Franx (1993ApJ...407..525V).

        Parameters
        ----------
        x : np.ndarray
            data values
        mu : float
            mean of x
        sigma : float
            standard deviation of x
        *h : float
            higher order coefficients

        Returns
        -------
        : np.ndarray
            function value of x
        """
        t = (x - mu) / sigma
        gauss = np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)
        hermite_polys = [H(t) for H in self._hermite_polys]
        gh_series = sum(_h * _H for _h, _H in zip(h, hermite_polys))
        # Eq. 18 of paper
        normalisation = sigma * (1 + np.sqrt(6) / 4 * h[1])
        return np.clip(gauss * (1 + gh_series) / normalisation, 1e-30, None)

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
        coeffs : list
            best fit coefficients (including mean and std)
        """
        if isinstance(data, UnitArr):
            data = data.view(np.ndarray)
        if len(data) == 0:
            _logger.warning("Data has length zero!")
            return 0.0, 0.0, 0.0, 0.0
        if np.any(np.isnan(data)):
            _logger.warning("Removing NaNs from data!")
            data = data[~np.isnan(data)]
        # the gauss hermite function is made to have the same mean and sigma as
        # the plain gaussian so compute them with faster estimates
        mu0 = np.nanmean(data)
        sigma0 = np.nanstd(data)
        coeffs = [mu0, sigma0]
        if self._hermite_order == 2:
            return coeffs

        def log_likelihood(pars):
            ll = -np.nansum(
                np.log(self.gauss_hermite_function(data, mu0, sigma0, *pars))
            )
            return ll

        num_h_coeffs = self._hermite_order - 2
        try:
            x0 = [0.0] * num_h_coeffs
            bounds = ([-1.0] * num_h_coeffs, [1.0] * num_h_coeffs)
            res = scipy.optimize.least_squares(
                log_likelihood, x0, loss="huber", bounds=bounds
            )
            h_coeffs = res.x
        except ValueError as err:
            _logger.warning(
                f"Unsuccessful fitting of Gauss-Hermite function for IFU bin - {err}"
            )
            h_coeffs = np.full(num_h_coeffs, np.nan)
        coeffs.extend(h_coeffs)
        return coeffs

    def binned_LOSV_statistics(self, p=4):
        """
        Generate the binned LOS statistics for the voronoi map.

        Parameters
        ----------
        p : int, optional
            Hermite polynomial order, by default 4
        """
        _logger.info(f"Binning {len(self.x):.2e} particles, Hermite order {p}")
        bin_index = list(range(self.max_voronoi_bin_index))

        bin_mass = scipy.ndimage.sum(
            self.m, labels=self._grid["particle_vor_bin_num"], index=bin_index
        )

        try:
            assert max(bin_index) > 1
        except AssertionError:
            _logger.exception("We only have one voronoi bin!", exc_info=True)
            raise

        if p < 2:
            _logger.warning("Gauss-Hermite series must include at least V and sigma")
            p = 2
        if p % 2 != 0:
            _logger.warning(
                f"Always fit matching symmetric and asymmetric components - increasing order from {p} to {p+1}"
            )
            p = p + 1
        self._hermite_order = p
        self._hermite_polys = [
            scipy.special.hermite(i, monic=True)
            for i in range(3, self._hermite_order + 1)
        ]

        fits = [None] * int(max(bin_index) + 1)
        assert len(np.unique(np.diff(bin_index))) == 1 and bin_index[0] == 0
        for idx in tqdm(bin_index, desc="Initialising Gauss Hermite fits"):
            fits[idx] = self.fit_gauss_hermite_distribution(
                self.vz[self._grid["particle_vor_bin_num"] == idx]
            )
        with TqdmCallback(desc="Fitting voronoi bins"):
            fits = dask.compute(*fits)
        bin_stats = np.array(fits)
        img_stats = bin_stats[self._grid["pixel_vor_bin_num"]]

        self._stats = dict(
            bin_V=bin_stats[:, 0],
            bin_sigma=bin_stats[:, 1],
            bin_mass=bin_mass,
            img_V=img_stats[..., 0],
            img_sigma=img_stats[..., 1],
        )

        for i in range(2, self._hermite_order):
            self._stats[f"bin_h{i+1}"] = bin_stats[:, i]
            self._stats[f"img_h{i+1}"] = img_stats[..., i]

    def plot_kinematic_maps(
        self,
        ax=None,
        moments=None,
        figsize=(7, 4.7),
        clims={},
        desat=False,
        cbar="adj",
        fontsize=None,
        cbar_kwargs={},
    ):
        """
        Plot the voronoi maps for a system.

        Parameters
        ----------
        vdat : dict
            voronoi values from baggins.analysis.voronoi_binned_los_V_statistics()
        ax : np.ndarray, optional
            numpy array of pyplot.Axes objects for plotting, by default None
        moments : str, optional
            moments as a 1-based hex string to plot, by default None
        figsize : tuple, optional
            figure size, by default (7,4.7)
        clims : dict, optional
            colour scale limits, by default None
        desat : bool, optional
            use a desaturated colour scheme, by default False
        cbar : str, optional
            how to add a colourbar to each map, can be "adj" for adjacent or "inset" for inset, by default "adj"
        fontsize : float or int, optional
            font size for colour bar label

        Returns
        -------
        ax : np.ndarray
            plotting axes
        """
        self._stat_is_calculated("V")
        # set the colour limits
        clims.setdefault("V", None)
        clims.setdefault("sigma", [None, None])
        for p in range(3, self._hermite_order + 1):
            clims.setdefault(f"h{p}", None)

        # set up the figure
        if moments is None:
            moments = "123456789ABCDEFG"[: self._hermite_order]
        if ax is None:
            num_cols = max(int((len(moments)) / 2), 1)
            fig, ax = plt.subplots(
                2, num_cols, sharex="all", sharey="all", figsize=figsize
            )
            for i in range(2):
                ax[i, 0].set_ylabel(r"$y/\mathrm{kpc}$")
            for i in range(num_cols):
                ax[-1, i].set_xlabel(r"$x/\mathrm{kpc}$")
        elif isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if desat:
            div_cols = colormaps.get("voronoi_div_desat")
            asc_cols = colormaps.get("voronoi_seq_desat")
        else:
            div_cols = colormaps.get("voronoi_div")
            asc_cols = colormaps.get("voronoi_seq")

        def _get_key_map_label(i):
            if i == 0:
                return "V", div_cols, r"$V/\mathrm{km}\,\mathrm{s}^{-1}$"
            elif i == 1:
                return "sigma", asc_cols, r"$\sigma/\mathrm{km}\,\mathrm{s}^{-1}$"
            else:
                return f"h{i+1}", div_cols, f"$h_{{{i+1}}}$"

        # get 0-based indexing for the moments in base 10
        moment_idxs = np.array([int(m, 17) - 1 for m in moments])
        try:
            assert np.all(moment_idxs < self._hermite_order)
        except AssertionError:
            _logger.exception(
                f"Requested Hermite order {np.max(moment_idxs)}, but Gauss-Hermite only fit to order {self._hermite_order}",
                exc_info=True,
            )
            raise

        for i, (m, axi) in enumerate(zip(moment_idxs, ax.flat)):
            statkey, cmap, label = _get_key_map_label(m)
            # plot the statistic
            cmap.set_bad(color="k")
            stat = self._stats[f"img_{statkey}"]
            if statkey != "sigma":
                norm = colors.CenteredNorm(vcenter=0, halfrange=clims[statkey])
            else:
                norm = colors.Normalize(*clims["sigma"])
            axi.set_aspect("equal")
            p1 = axi.imshow(
                stat,
                interpolation="nearest",
                origin="lower",
                extent=self.extent,
                cmap=cmap,
                norm=norm,
            )
            label = cbar_kwargs.pop("label", label)
            if "horizontal_alignment" in cbar_kwargs:
                cbar_kwargs["ha"] = cbar_kwargs["horizontal_alignment"]
            ha = cbar_kwargs.pop("ha", "right")
            if cbar == "adj":
                divider = make_axes_locatable(axi)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cb = plt.colorbar(p1, cax=cax, **cbar_kwargs)
                cb.set_label(label=label, size=fontsize)
            elif cbar == "inset":
                alignment = 0.4 if ha == "right" else 0.05
                cax = axi.inset_axes([alignment, 0.94, 0.55, 0.04])
                cax.set_xticks([])
                cax.set_yticks([])
                cax.patch.set_alpha(0)
                cb = plt.colorbar(p1, cax=cax, orientation="horizontal", **cbar_kwargs)
                cb.set_label(label=label, size=fontsize)
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
            x = float(x)
            y = float(y)
        except TypeError as err:
            _logger.exception(
                f"Coordinates must be convertible to a float. {err}", exc_info=True
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
        density = kwargs.pop("density", True)
        h = ax.hist(sample, density=density, **kwargs)

        # add the LOSD over the top
        _v = np.linspace(h[1][0], h[1][-1], 1000)
        if density:
            scale = 1
        else:
            scale = scipy.integrate.trapezoid(h[0], get_histogram_bin_centres(h[1]))
        ax.plot(
            _v,
            scale
            * self.gauss_hermite_function(
                _v,
                self.img_V[indx, indy],
                self.img_sigma[indx, indy],
                *[
                    self.stats[f"img_h{i}"][indx, indy]
                    for i in range(3, 3 + len(self._hermite_polys))
                ],
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

    def dump_to_dict(self):
        """
        Dump the minimum amount of data to a dict that can then be used to reinitialise the class

        Returns
        -------
        d : dict
            necessary data
        """
        d = {}
        d.update(self._grid)
        d.update(self.stats)
        d["extent"] = self.extent
        d["vz"] = self.vz
        return d

    @classmethod
    def load_from_dict(cls, d):
        """
        Initiate class from a dictionary. Not all keys will have available data.

        Parameters
        ----------
        d : dict
            Dictionary to load data from

        Returns
        -------
        C : VoronoiKinematics
            initiated class
        """
        C = cls(x=[], y=[], V=[], m=[])
        grid_keys = [
            "particle_vor_bin_num",
            "pixel_vor_bin_num",
            "x_bin",
            "y_bin",
            "xedges",
            "yedges",
        ]
        stat_keys = list(set(d.keys()).difference(set(grid_keys)))
        try:
            C._hermite_order = max(
                [int(v.replace("bin_h", "")) for v in stat_keys if "bin_h" in v]
            )
        except ValueError:
            # there are no h moments
            C._hermite_order = 2
        C._hermite_polys = [
            scipy.special.hermite(i, monic=True) for i in range(3, C._hermite_order + 1)
        ]
        C._grid = {}
        C._stats = {}
        for k in grid_keys:
            try:
                C._grid[k] = d[k]
            except KeyError:
                C._grid[k] = None
        for k in stat_keys:
            try:
                C._stats[k] = d[k]
            except KeyError:
                C._stats[k] = None
        C._extent = d["extent"]
        try:
            C.vz = d["vz"]
        except KeyError:
            _logger.warning(
                "File was created before pseudo-particle velocity information was saved: only LOSVD information accessible."
            )
            C.vz = None
        return C


def unify_IFU_colour_scheme(vor_list):
    clims = None
    for v in vor_list:
        voronoi = VoronoiKinematics.load_from_dict(v)
        if clims is None:
            clims = voronoi.get_colour_limits()
        else:
            _clims = voronoi.get_colour_limits()
            for k in _clims:
                if k == "sigma":
                    clims[k][0] = np.min([clims[k][0], _clims[k][0]])
                    clims[k][1] = np.max([clims[k][1], _clims[k][1]])
                else:
                    clims[k] = max([clims[k], _clims[k]])
    return clims
