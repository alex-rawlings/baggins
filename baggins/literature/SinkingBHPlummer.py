import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf
import ketjugw
from baggins.general.units import Gyr, kpc
from baggins.mathematics import radial_separation
from baggins.utils import get_ketjubhs_in_dir

__all__ = ["SinkingBHPlummer"]


class SinkingBHPlummer:
    def __init__(self, MBH, Mstar, a):
        """
        Orbit-averaged dynamical friction sinking of one or more BHs on circular
        orbits within a Plummer sphere of stars. Each BH is assumed to interact
        only with the background stellar distribution (i.e. BH-BH interactions
        are neglected), so BHs of different mass sink independently.

        Parameters
        ----------
        MBH : float or array-like
            mass of the sinking BH(s). A scalar sinks a single BH; an array-like
            sinks one BH per entry, each starting at the virial radius.
        Mstar : float
            total mass of the Plummer sphere of stars
        a : float
            Plummer scale radius
        """
        self.MBH = np.atleast_1d(np.asarray(MBH, dtype=float))
        self.Mstar = Mstar
        self.a = a * kpc

        # derived quantities
        self.rvir = 16 / (3 * np.pi) * self.a
        self.bmax = 2 * self.rvir
        self.t_dyn = self.bmax / np.sqrt(self.Mstar / self.bmax)
        self.b90 = self.MBH / self.sigma(self.rvir) ** 2
        self.logL = np.log(self.bmax / self.b90)

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
        Determine sinking velocity for each BH independently.

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
        r0 = np.full_like(self.MBH, self.rvir)
        sol = solve_ivp(self.drdt, (t0, tf), r0, t_eval=self.ts)
        self.analytical_sep = sol.y

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
            rf"$M_\bullet={np.array2string(self.MBH, formatter={'float_kind':lambda x: '%.2e' % x})}\,\mathrm{{M}}_\odot, M_\star={self.Mstar:.2e}\,\mathrm{{M}}_\odot, a={self.a/kpc:.2e}\,\mathrm{{kpc}}$"
        )
        label = kwargs.pop("label", None)
        kwargs.setdefault("c", "k")
        kwargs.setdefault("lw", 3)
        for i, sep in enumerate(self.analytical_sep):
            if label is None:
                this_label = (
                    "Analytical"
                    if len(self.MBH) == 1
                    else rf"Analytical ($M_\mathrm{{BH}}={self.MBH[i]:.3g}$)"
                )
            else:
                this_label = label if len(self.MBH) == 1 else f"{label} ({i})"
            ax.plot(self.ts / Gyr, sep / kpc, label=this_label, **kwargs)
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
        ketju_files = get_ketjubhs_in_dir(simdir)
        for f in ketju_files:
            bh = list(ketjugw.load_hdf5(f).values())[0]
            ax.plot(bh.t / Gyr, radial_separation(bh.x / kpc), **kwargs)
        ax.legend()
        return ax
