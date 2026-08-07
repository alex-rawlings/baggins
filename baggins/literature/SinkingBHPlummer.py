import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf
from tqdm import tqdm
import ketjugw
import pygad
from baggins.env_config import _cmlogger
from baggins.general import convert_gadget_time
from baggins.general.units import Gyr, kpc
from baggins.mathematics import radial_separation
from baggins.utils import get_ketjubhs_in_dir, get_snapshots_in_dir

__all__ = ["SinkingBHPlummer"]

_logger = _cmlogger.getChild(__name__)


class SinkingBHPlummer:
    def __init__(self, MBH, Mstar, a, bmax_fac=2):
        """
        Orbit-averaged dynamical friction sinking of a BH on a circular orbit
        within a Plummer sphere of stars, starting at the virial radius.

        Parameters
        ----------
        MBH : float
            mass of the sinking BH
        Mstar : float
            total mass of the Plummer sphere of stars
        a : float
            Plummer scale radius
        """
        self.MBH = float(MBH)
        self.Mstar = Mstar
        self.a = a * kpc

        # derived quantities
        self.rvir = 16 / (3 * np.pi) * self.a
        self.bmax = bmax_fac * self.rvir
        self.t_dyn = self.bmax / np.sqrt(self.Mstar / self.bmax)
        self.b90 = self.MBH / self.sigma(self.rvir) ** 2
        self.logL = np.log(self.bmax / self.b90)
        _logger.debug(f"bmax:{self.bmax / kpc:.3e} kpc")
        _logger.debug(f"b90: {self.b90 / kpc:.3e} kpc")
        _logger.debug(f"Coulomb log: {self.logL:.3e}")

        # place holders
        self.ts = None
        self.analytical_sep = None

    def density(self, r):
        """
        Stellar density.

        Parameters
        ----------
        r : float or array-like
            radius to evaluate at

        Returns
        -------
        : float or array-like
            density
        """
        return (
            3 * self.Mstar / (4 * np.pi * self.a**3) * (1 + r**2 / self.a**2) ** (-2.5)
        )

    def sigma(self, r):
        """
        Stellar velocity dispersion.

        Parameters
        ----------
        r : float or array-like
            radius to evaluate at

        Returns
        -------
        : float or array-like
            velocity dispersion
        """
        return np.sqrt(self.Mstar / (6 * np.sqrt(r**2 + self.a**2)))

    def drdt(self, _, r):
        """
        Determine sinking velocity of the BH.

        Parameters
        ----------
        _ :
            place holder for solver compatability
        r : array-like
            radii

        Returns
        -------
        : array-like
            sinking velocity due to dynamical friction
        """
        v_circ = np.sqrt(self.Mstar * (1 + self.a**2 / r**2) ** (-1.5) / r)
        X = v_circ / (np.sqrt(2) * self.sigma(r))
        chi = erf(X) - 2 * X * np.exp(-(X**2)) / np.sqrt(np.pi)
        return (
            -8
            * np.pi
            * self.logL
            * self.density(r)
            * chi
            * self.MBH
            * r
            / (v_circ**3 * (1 + 3 / (1 + r**2 / self.a**2)))
        )

    def evolve(self, t0=0, tf=3, steps=200):
        """
        Solve the infall trajectory of the sinking BH.

        Parameters
        ----------
        t0 : float, optional
            initial time (Gyr), by default 0
        tf : float, optional
            final time (Gyr), by default 3
        steps : int, optional
            number of evaluation steps, by default 200
        """
        tf = tf * Gyr
        self.ts = np.linspace(t0, tf, steps)
        sol = solve_ivp(self.drdt, (t0, tf), [self.rvir], t_eval=self.ts)
        self.analytical_sep = sol.y

    def plot_bmax_range(
        self, ax, bmax_fac_range=(1, 3), t0=0, tf=3, steps=200, **kwargs
    ):
        """
        Shade the region spanned by the analytical trajectory as ``bmax_fac``
        is varied, to indicate the sensitivity of the Coulomb logarithm to
        the (poorly constrained) choice of maximum impact parameter.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            plotting axes
        bmax_fac_range : tuple of float, optional
            (lower, upper) bmax_fac bounding the shaded region, by default
            (1, 3)
        t0 : float, optional
            initial time (Gyr), by default 0
        tf : float, optional
            final time (Gyr), by default 3
        steps : int, optional
            number of evaluation steps, by default 200

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axes
        """
        bmax_fac_lo, bmax_fac_hi = bmax_fac_range
        # smaller bmax_fac -> smaller Coulomb log -> weaker friction -> slower sink (larger r)
        slow_sink = SinkingBHPlummer(
            self.MBH, self.Mstar, self.a / kpc, bmax_fac=bmax_fac_lo
        )
        fast_sink = SinkingBHPlummer(
            self.MBH, self.Mstar, self.a / kpc, bmax_fac=bmax_fac_hi
        )
        slow_sink.evolve(t0=t0, tf=tf, steps=steps)
        fast_sink.evolve(t0=t0, tf=tf, steps=steps)
        kwargs.setdefault("ec", "none")
        kwargs.setdefault("fc", "gray")
        kwargs.setdefault("alpha", 0.6)
        ax.fill_between(
            slow_sink.ts / Gyr,
            fast_sink.analytical_sep[0] / kpc,
            slow_sink.analytical_sep[0] / kpc,
            **kwargs,
        )
        return ax

    def plot(self, ax=None, legend=True, **kwargs):
        """
        Plot analytical trajectory.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        legend : bool, optional
            plot legend, by default True

        Returns
        -------
        ax : matplotlib.axes.Axes
                    plotting axes, by default None
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$t/\mathrm{Gyr}$")
            ax.set_ylabel(r"$r/\mathrm{kpc}$")
        else:
            fig = ax.get_figure()
        fig.suptitle(
            rf"$M_\bullet={self.MBH:.3g}\,\mathrm{{M}}_\odot, M_\star={self.Mstar:.2e}\,\mathrm{{M}}_\odot, a={self.a/kpc:.2e}\,\mathrm{{kpc}}$"
        )
        label = kwargs.pop("label", "Analytical")
        kwargs.setdefault("c", "k")
        kwargs.setdefault("lw", 3)
        ax.plot(self.ts / Gyr, self.analytical_sep[0] / kpc, label=label, **kwargs)
        if legend:
            ax.legend()
        return ax

    def plot_simulation(self, simdir, ax, **kwargs):
        """
        Plot simulation trajectory over the analytical curve.

        Parameters
        ----------
        simdir : str
            simulation output directory
        ax : matplotlib.axes.Axes
            plotting axes, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        """
        has_ketju = True
        label = kwargs.setdefault("label", "")
        try:
            ketju_file = get_ketjubhs_in_dir(simdir)[0]
            label += " Ketju"
        except IndexError:
            _logger.warning("BH position will be taken from snapshots!")
            has_ketju = False
            label += " Gadget"
        kwargs["label"] = label
        snapfiles = get_snapshots_in_dir(simdir)
        N = len(snapfiles)
        t = np.full(N, np.nan)
        xcom = np.full((N, 3), np.nan)
        snap_bh = np.full_like(xcom, np.nan)
        com_kwargs = {"center": np.zeros(3), "R": self.rvir, "stop_N": 100}
        for i, snapfile in tqdm(enumerate(snapfiles), desc="Finding centres", total=N):
            snap = pygad.Snapshot(snapfile, physical=True)
            try:
                assert len(snap.bh) == 1
            except AssertionError:
                _logger.exception(
                    f"Only 1 BH permitted in snapshot, not {len(snap.bh)}",
                    exc_info=True,
                )
                raise
            t[i] = convert_gadget_time(snap)
            try:
                xcom[i, :] = pygad.analysis.shrinking_sphere(
                    snap.stars, **com_kwargs
                ).view(np.ndarray)
            except AttributeError:
                xcom[i, :] = pygad.analysis.shrinking_sphere(
                    snap.dm, **com_kwargs
                ).view(np.ndarray)
            snap_bh[i, :] = snap.bh["pos"]
            del snap
            pygad.gc_full_collect()
        rcom = radial_separation(xcom)
        _logger.info(f"CoM wanders {np.min(rcom):.2e} kpc - {np.max(rcom):.2e} kpc")
        if has_ketju:
            bh = list(ketjugw.load_hdf5(ketju_file).values())[0]
            xcom_interp = np.full((len(bh), 3), np.nan)
            for i in range(3):
                xcom_interp[:, i] = np.interp(
                    bh.t / Gyr, t, xcom[:, i], left=np.nan, right=np.nan
                )
            ax.plot(bh.t / Gyr, radial_separation(xcom_interp - bh.x / kpc), **kwargs)
        else:
            ax.plot(t, radial_separation(snap_bh - xcom), marker=".", **kwargs)
        ax.set_ylim(None, 2 * self.a / kpc)
        ax.legend()

        return ax
