import os.path
from functools import cached_property
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import griddata, RectBivariateSpline
import pygad
from baggins.env_config import _cmlogger
from baggins.analysis.analyse_snap import hardening_radius, influence_radius
from baggins.mathematics import radial_separation, fit_ellipse, eccentricity
from baggins.plotting import extract_contours_from_plot
from baggins.utils import get_snapshots_in_dir

_logger = _cmlogger.getChild(__name__)

__all__ = ["PotentialFitter"]


class PotentialFitter:
    def __init__(self, snapfile, oblate=False, major_axis="x"):
        self.a_hard = None
        if os.path.isfile(snapfile):
            self.snapfile = snapfile
        else:
            self.snapfile = self._get_hard_binary_snap(get_snapshots_in_dir(snapfile))
        _logger.info(f"Fitting potential to {self.snapfile}")
        self.oblate = oblate
        self.major_axis = major_axis
        snap = pygad.Snapshot(self.snapfile, physical=True)
        pygad.Translation(-pygad.analysis.center_of_mass(snap.bh)).apply(snap)
        if self.a_hard is None:
            # called only if a file was passed
            rinfl = influence_radius(snap, combined=True)
            self.a_hard = hardening_radius(
                snap.bh["mass"], list(rinfl.values())[0]
            ).view(np.ndarray)
        bh_sep = pygad.utils.geo.dist(np.diff(snap.bh["pos"], axis=0))
        _logger.info(f"BH separation is {bh_sep}")
        _logger.debug(f"Hardening radius is {self.a_hard}")

        # grid the potential from those stars that are in the plane
        # perpendicular to the angular momentum vector of the BH binary
        L = np.cross(np.diff(snap.bh["pos"], axis=0), np.diff(snap.bh["vel"], axis=0))
        Lhat = L / np.linalg.norm(L)
        self.stars = snap.stars[abs(np.dot(snap.stars["pos"], Lhat[0])) < 1e-2]
        assert len(self.stars) > 10
        self.ellipse_angle = None
        self._fitted_ellipse = None

    @property
    def e_spheroid(self):
        return self._e_spheroid

    @e_spheroid.setter
    def e_spheroid(self, v):
        self._e_spheroid = v
        for k in ("_e2s", "A1", "A3"):
            self.__dict__.pop(k, None)

    @cached_property
    def _e2s(self):
        return self.e_spheroid**2

    @cached_property
    def A1(self):
        if self.oblate:
            return (
                np.sqrt(1 - self._e2s)
                / self._e2s
                * (
                    np.arcsin(self.e_spheroid) / self.e_spheroid
                    - np.sqrt(1 - self._e2s)
                )
            )
        else:
            return (
                (1 - self._e2s)
                / self._e2s
                * (
                    1 / (1 - self._e2s)
                    - 1
                    / (2 * self.e_spheroid)
                    * np.log((1 + self.e_spheroid) / (1 - self.e_spheroid))
                )
            )

    @cached_property
    def A3(self):
        if self.oblate:
            return (
                2
                * np.sqrt(1 - self._e2s)
                / self._e2s
                * (
                    1 / np.sqrt(1 - self._e2s)
                    - np.arcsin(self.e_spheroid) / self.e_spheroid
                )
            )
        else:
            return (
                2
                * (1 - self._e2s)
                / self._e2s
                * (
                    1
                    / (2 * self.e_spheroid)
                    * np.log((1 + self.e_spheroid) / (1 - self.e_spheroid))
                    - 1
                )
            )

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def R(self):
        return radial_separation(
            np.stack((self.X.flatten(), self.Y.flatten()), axis=-1)
        )

    @property
    def pot(self):
        return self._pot

    def __str__(self):
        s = f"{self.__class__.__name__}: e_spheroid={self.e_spheroid:.3f}, oblate={self.oblate}, phi={np.degrees(self.ellipse_angle):.3f}"
        return s

    def _get_hard_binary_snap(self, sl):
        a_hard = np.full(len(sl), np.nan)
        for i, sf in enumerate(sl):
            snap = pygad.Snapshot(sf, physical=True)
            pygad.Translation(-pygad.analysis.center_of_mass(snap.bh)).apply(snap)
            rinfl = influence_radius(snap, combined=True)
            try:
                a_hard[i] = hardening_radius(
                    snap.bh["mass"], list(rinfl.values())[0]
                ).view(np.ndarray)
            except AssertionError:
                self.a_hard = a_hard[i - 1]
                return sl[i - 1]
            if pygad.utils.geo.dist(np.diff(snap.bh["pos"], axis=0)) < a_hard[i]:
                snap.delete_blocks()
                pygad.gc_full_collect()
                self.a_hard = a_hard[i - 1]
                return sl[i - 1]
        else:
            snap.delete_blocks()
            pygad.gc_full_collect()
            self.a_hard = a_hard[-1]
            return sl[-1]

    def _get_last_snap_with_binary(self, sl):
        for i, sf in enumerate(sl):
            with h5py.File(sf, "r") as f:
                if len(f["/PartType5/ParticleIDs"][:]) < 2:
                    return sl[i - 1]
        else:
            _logger.warning(
                "All snapshots have at least two BHs! Returning last snapshot..."
            )
            return sl[-1]

    def _set_default_contour_kwargs(self, dt, da):
        for d in (dt, da):
            # for the true potential
            d.setdefault(
                "levels",
                lambda p: np.sort(np.geomspace(np.nanmin(p), np.nanmax(p), 20)),
            )
            d.setdefault("linestyles", "-")
            d.setdefault("linewidths", 0.5)
        dt.setdefault("colors", "k")
        da.setdefault("colors", "r")
        return dt, da

    def _rotate(self, x, y, phi):
        xr = np.cos(phi) * x - np.sin(phi) * y
        yr = np.cos(phi) * y + np.sin(phi) * x
        return xr, yr

    def fit_potential(self, extent=1, levels=None):
        self._X, self._Y = np.meshgrid(*2 * [np.linspace(-extent, extent, 400)])
        self._pot = np.array(
            griddata(self.stars["pos"][:, [0, 2]], self.stars["Epot"], (self.X, self.Y))
        )
        if levels is None:
            # take the median potential at the hardening radius
            pot_interp = RectBivariateSpline(
                *2 * [np.linspace(-extent, extent, self._X.shape[0])], self.pot
            )
            pot_vals = np.full(100, np.nan)
            r = 2 * self.a_hard
            for i, t in enumerate(np.linspace(0, 2 * np.pi, len(pot_vals))):
                pot_vals[i] = pot_interp(r * np.cos(t), r * np.sin(t))
            levels = [np.nanmedian(pot_vals)]
        elif levels == "median":
            levels = [np.nanmedian(self.pot)]
        else:
            try:
                assert isinstance(levels, (list, int))
            except AssertionError:
                _logger.exception(
                    f"levels must be a list, an int, or 'median', not {levels}",
                    exc_info=True,
                )
                raise
        _logger.debug(f"Using level {levels}")
        p = plt.contour(self.X, self.Y, self.pot, levels=levels)
        plt.close()
        xc, yc = extract_contours_from_plot(p=p)
        try:
            assert xc
        except AssertionError:
            _logger.exception(
                "No contours fit! Try changing 'level' parameter.", exc_info=True
            )
            raise
        e_spheroid_vals = []
        phi_vals = []
        for i, (x, y) in enumerate(zip(xc, yc)):
            ellip, a, b, phi = fit_ellipse(x, y)
            e_spheroid_vals.append(eccentricity(a, b))
            phi_vals.append(phi)
        if len(e_spheroid_vals) == 1:
            self.e_spheroid = e_spheroid_vals[0]
            self.ellipse_angle = phi + (np.pi / 2 if self.major_axis == "y" else 0)
            self._fitted_ellipse = ellip
        return e_spheroid_vals, phi_vals

    def plot(self, ax=None, tpkwargs={}, apkwargs={}):
        if ax is None:
            fig, ax = plt.subplots()
        self._set_default_contour_kwargs(tpkwargs, apkwargs)
        Xr, Yr = self._rotate(self.X, self.Y, -self.ellipse_angle)
        if callable(tpkwargs["levels"]):
            tpkwargs["levels"] = tpkwargs["levels"](self.pot)
        cpr = ax.contour(Xr, Yr, self.pot, **tpkwargs)
        if self._fitted_ellipse is not None:
            ax.plot(
                *self._rotate(*self._fitted_ellipse, -self.ellipse_angle),
                lw=1,
                c="tab:purple",
                label="Fitted ellipse",
            )

        potA = self.A3 * self.X**2 + self.A1 * self.Y**2
        if callable(apkwargs["levels"]):
            apkwargs["levels"] = apkwargs["levels"](potA)
        cpa = ax.contour(self.X, self.Y, potA, label="Analytic", **apkwargs)
        ax.set_title(f"es = {self.e_spheroid:.3f}")
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        for cp, lab in zip((cpr, cpa), ("True", "Analytic")):
            _cpc = cp.collections[0]
            proxyline = Line2D(
                [0], [0], c=_cpc.get_edgecolor(), lw=_cpc.get_linewidth()
            )
            handles.append(proxyline)
            labels.append(lab)
        ax.legend(handles, labels)
        return ax

    def plot_potential_1D(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel("r/kpc")
            ax.set_ylabel("-Potential")
        ax.loglog(self.R, -self.pot.flatten(), marker=".", ls="", markersize=1)
        return ax
