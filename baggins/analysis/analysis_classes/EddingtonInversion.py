import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

__all__ = ["EddingtonInversion"]


class EddingtonInversion:
    def __init__(self, snap, n_grid: int = 256, spline_smooth: float = None):
        """
        Compute the isotropic phase-space DF f(E) from particle snapshot data via
        Eddington inversion with an Abel-transform acceleration.

        Parameters
        ----------
        snap : dict-like
            Snapshot dictionary with keys:
            'pos'  : (N, 3) array of positions  [any length unit]
            'vel'  : (N, 3) array of velocities [corresponding velocity unit]
            'mass' : (N,)   array of particle masses
            'pot'  : (N,)   array of gravitational potential at each particle
                            (negative values, already centred on stellar CoM)
        n_grid : int
            Number of grid points in Psi-space used for the Abel-transform
            integration (default: 256). Larger values give smoother f(E) at the
            cost of compute time.
        spline_smooth : float or None
            Smoothing factor passed to 'scipy.interpolate.UnivariateSpline' when fitting rho(Psi).  Larger values allow more deviation from the data in exchange for a smoother (and hence better-conditioned) second derivative.
            ``None`` uses SciPy's default heuristic (sum of squared residuals ~= N). A value of 0 forces interpolation through every point, which is rarely desirable (default: None).
        """
        self.snap = snap
        self.n_grid = int(n_grid)
        self.spline_smooth = spline_smooth

        # ---- cached per-particle quantities (filled lazily) ----
        self._r = None  # (N,) radii
        self._v2 = None  # (N,) squared speeds
        self._psi = None  # (N,) relative potential at each particle
        self._energy = None  # (N,) relative binding energy E = Psi - 0.5*v^2

        # ---- cached grid quantities ----
        self._psi_grid = None  # (n_grid,) uniform Psi grid
        self._rho_grid = None  # (n_grid,) density on Psi grid
        self._rho_spline = (
            None  # UnivariateSpline fit to rho(Psi), reused for derivatives
        )
        self._f_grid = None  # (n_grid,) f(E) on Psi grid
        self._f_interp = None  # callable interpolator f(E)

        # ---- scalar bookkeeping ----
        self._psi_max = None  # = -pot_min  (reference potential)

    # ------------------------------------------------------------------
    # Step 1 – basic kinematic quantities
    # ------------------------------------------------------------------

    def compute_radii(self) -> np.ndarray:
        """
        Compute and cache particle radii from the CoM (assumed origin).

        Returns
        -------
        r : (N,) ndarray
        """
        if self._r is None:
            pos = np.asarray(self.snap["pos"])
            self._r = np.sqrt(np.einsum("ij,ij->i", pos, pos))
        return self._r

    def compute_speeds_squared(self) -> np.ndarray:
        """
        Compute and cache squared particle speeds.

        Returns
        -------
        v2 : (N,) ndarray
        """
        if self._v2 is None:
            vel = np.asarray(self.snap["vel"])
            self._v2 = np.einsum("ij,ij->i", vel, vel)
        return self._v2

    # ------------------------------------------------------------------
    # Step 2 – relative potential and binding energy
    # ------------------------------------------------------------------

    def compute_relative_potential(self) -> np.ndarray:
        """
        Convert the raw gravitational potential to the *relative* potential

            Psi_i = Phi_max - Phi_i

        so that Psi >= 0 everywhere and Psi -> 0 at the boundary.  The
        reference value Phi_max is the maximum (least-negative) potential
        among all particles, approximating Phi(r -> infinity) = 0.

        Returns
        -------
        psi : (N,) ndarray  – relative potential per particle
        """
        if self._psi is None:
            pot = np.asarray(self.snap["pot"])
            # Phi is negative; Phi_max ~ 0 at the edge of the system.
            self._psi_max = float(np.max(pot))
            self._psi = self._psi_max - pot  # >= 0 for all particles
        return self._psi

    def compute_energies(self) -> np.ndarray:
        """
        Compute and cache the relative specific binding energy per particle:

            E_i = Psi_i - 0.5 * v_i^2

        Bound particles have E > 0.

        Returns
        -------
        energy : (N,) ndarray
        """
        if self._energy is None:
            psi = self.compute_relative_potential()
            v2 = self.compute_speeds_squared()
            self._energy = psi - 0.5 * v2
        return self._energy

    # ------------------------------------------------------------------
    # Step 3 – density profile rho(Psi) on a uniform grid
    # ------------------------------------------------------------------

    def compute_density_profile(self) -> tuple:
        """
        Estimate the mass density as a function of the relative potential
        rho(Psi) on a uniform Psi grid.

        The approach has two clearly separated stages:

        Stage A – rho(r) in radial shells
          Particles are binned by radius using equal-number (quantile) shells so
          that every shell has comparable Poisson noise.  The shell volume is
          V_shell = (4/3) pi (r_outer^3 - r_inner^3), computed exactly from the
          bin edges, with no dependence on the potential.

        Stage B – map r -> Psi, then resample onto a uniform Psi grid
          The mean potential in each radial shell gives a smooth, monotone
          Psi(r) relation.  A spline fit to this relation is used to convert
          the radial bin centres to Psi values.  The resulting (Psi_i, rho_i)
          pairs are then resampled onto a uniform Psi grid via a second spline,
          ready for differentiation in the next step.

        This separation means the density estimate is never corrupted by scatter
        in the r-Psi mapping: shell volumes are exact and the potential is only
        used to establish the Psi axis, not to define mass bins.

        Returns
        -------
        psi_grid : (n_grid,) ndarray
        rho_grid : (n_grid,) ndarray  [same mass / length^3 units as input]
        """
        if self._rho_grid is not None:
            return self._psi_grid, self._rho_grid

        psi = self.compute_relative_potential()
        r = self.compute_radii()
        mass = np.asarray(self.snap["mass"])

        # ------------------------------------------------------------------
        # Stage A: rho(r) — bin in radius, compute exact shell volumes
        # ------------------------------------------------------------------
        # Equal-number radial bins so every shell has similar particle count.
        r_bin_edges = np.percentile(r, np.linspace(0, 100, self.n_grid + 1))
        # Ensure strictly increasing edges (duplicates can occur in very
        # concentrated systems where many particles share the same radius).
        r_bin_edges = np.unique(r_bin_edges)
        n_bins = len(r_bin_edges) - 1

        mass_in_shell, _ = np.histogram(r, bins=r_bin_edges, weights=mass)
        shell_vol = (4.0 / 3.0) * np.pi * (r_bin_edges[1:] ** 3 - r_bin_edges[:-1] ** 3)
        shell_vol = np.maximum(shell_vol, np.finfo(float).tiny)

        rho_r = mass_in_shell / shell_vol  # (n_bins,)

        # ------------------------------------------------------------------
        # Stage B: build mean Psi(r) per shell, then resample rho onto Psi grid
        # ------------------------------------------------------------------
        # Mean potential in each radial shell (mass-weighted for robustness).
        psi_mean = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (r >= r_bin_edges[i]) & (r < r_bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            w = mass[mask]
            psi_mean[i] = np.average(psi[mask], weights=w)
        # print(np.unique(psi_mean))

        # Psi increases inward (high Psi = deep potential well = small r).
        # Ensure the Psi(r_mid) relation is monotone by sorting on Psi.
        order = np.argsort(psi_mean)
        psi_sorted = psi_mean[order]
        rho_sorted = rho_r[order]

        # Remove any remaining non-monotone duplicates.
        _, unique_idx = np.unique(psi_sorted, return_index=True)
        psi_sorted = psi_sorted[unique_idx]
        rho_sorted = rho_sorted[unique_idx]

        # Uniform Psi grid spanning the range of well-sampled shells.
        psi_min_pos = max(float(psi_sorted[0]), 1e-10 * float(psi_sorted[-1]))
        psi_max = float(psi_sorted[-1])
        self._psi_grid = np.linspace(psi_min_pos, psi_max, self.n_grid)

        # Resample rho onto the uniform Psi grid with a smoothing spline.
        rho_of_psi = UnivariateSpline(
            psi_sorted,
            rho_sorted,
            k=3,
            s=self.spline_smooth,
            ext="const",
        )
        self._rho_grid = np.maximum(rho_of_psi(self._psi_grid), 0.0)

        return self._psi_grid, self._rho_grid

    # ------------------------------------------------------------------
    # Step 4 – second derivative d^2 rho / d Psi^2
    # ------------------------------------------------------------------

    def compute_rho_second_derivative(self) -> np.ndarray:
        """
        Compute d^2 rho / d Psi^2 on the uniform Psi grid by fitting a cubic
        smoothing spline to rho(Psi) and evaluating its analytic second derivative.

        This is significantly more robust than finite differences on noisy
        particle-histogram data: the spline jointly smooths *and* differentiates,
        avoiding the amplification of high-frequency noise that plagues repeated
        finite-difference stencils.

        The fitted spline is cached in ``self._rho_spline`` so that the same
        representation can be reused by ``compute_distribution_function`` to
        evaluate the boundary term d rho / d Psi |_{Psi=0} without an additional
        finite-difference approximation.

        Returns
        -------
        d2rho_dpsi2 : (n_grid,) ndarray
        """
        psi_grid, rho_grid = self.compute_density_profile()

        # Fit a cubic smoothing spline (k=4 would give a C3 spline, but k=3
        # is the standard choice; its second derivative is piecewise linear and
        # C1-continuous, which is sufficient for the Abel integral).
        if self._rho_spline is None:
            self._rho_spline = UnivariateSpline(
                psi_grid,
                rho_grid,
                k=4,  # quartic: second derivative is smooth
                s=self.spline_smooth,  # smoothing factor (None = heuristic)
                ext="const",  # extrapolate as constant outside range
            )

        d2rho = self._rho_spline.derivative(n=2)(psi_grid)
        return d2rho

    # ------------------------------------------------------------------
    # Step 5 – Eddington inversion via Abel transform
    # ------------------------------------------------------------------

    def compute_distribution_function(self) -> tuple:
        """
        Evaluate the isotropic DF f(E) via Eddington's inversion formula:

            f(E) = 1/(sqrt(8) * pi^2) * d/dE [
                integral_0^E  (d rho / d Psi) / sqrt(E - Psi) d Psi
            ]

        Equivalently (integrating by parts once):

            f(E) = 1/(sqrt(8) * pi^2) * [
                integral_0^E  (d^2 rho / d Psi^2) / sqrt(E - Psi) d Psi
                + (d rho / d Psi)|_{Psi=0} / sqrt(E)
            ]

        The integral is the Abel transform of d^2 rho / d Psi^2, evaluated
        efficiently on the same grid by cumulative trapezoidal integration.

        For each grid point E_j = Psi_j, we compute:

            I(E_j) = sum_{k < j}  g(Psi_k) / sqrt(E_j - Psi_k) * d Psi

        where g = d^2 rho / d Psi^2.  The singularity at Psi = E is handled
        by analytically integrating the last half-interval:

            integral_{E-h}^{E}  g(E) / sqrt(E-Psi) d Psi  ~  2 * g(E) * sqrt(h)

        Returns
        -------
        psi_grid : (n_grid,) ndarray  – energy axis (= Psi grid)
        f_grid   : (n_grid,) ndarray  – f(E) values (zero for unbound energies)
        """
        if self._f_grid is not None:
            return self._psi_grid, self._f_grid

        psi_grid, rho_grid = self.compute_density_profile()
        d2rho = self.compute_rho_second_derivative()  # also populates self._rho_spline

        n = len(psi_grid)
        dpsi = psi_grid[1] - psi_grid[0]

        # First derivative at Psi = 0 from the spline (exact, not finite-difference).
        drho_dpsi_0 = float(self._rho_spline.derivative(n=1)(psi_grid[0]))

        # Abel integral  I(E_j) = integral_0^{E_j} d2rho(Psi)/sqrt(E_j - Psi) dPsi
        # Evaluated with the singularity-corrected trapezoidal rule.
        abel = np.zeros(n)
        for j in range(1, n):
            E_j = psi_grid[j]
            psi_k = psi_grid[:j]
            g_k = d2rho[:j]
            # Regular trapezoidal contribution (interior points, non-singular).
            weights = 1.0 / np.sqrt(E_j - psi_k)
            # Avoid integrating the last point where denominator -> 0;
            # handle it analytically: 2 * g(E) * sqrt(dpsi).
            interior = np.trapz(g_k * weights, psi_k) if j > 1 else 0.0
            # Analytic singular cap for the last sub-interval [Psi_{j-1}, Psi_j].
            singular_cap = 2.0 * d2rho[j] * np.sqrt(dpsi)
            abel[j] = interior + singular_cap

        # Boundary term.
        boundary = np.where(psi_grid > 0, drho_dpsi_0 / np.sqrt(psi_grid), 0.0)

        prefactor = 1.0 / (np.sqrt(8.0) * np.pi**2)
        self._f_grid = prefactor * (abel + boundary)

        # Enforce physical constraint: f(E) >= 0.
        self._f_grid = np.maximum(self._f_grid, 0.0)

        # Build interpolator for arbitrary E queries.
        self._f_interp = interp1d(
            psi_grid, self._f_grid, kind="linear", bounds_error=False, fill_value=0.0
        )

        return self._psi_grid, self._f_grid

    # ------------------------------------------------------------------
    # Step 6 – public evaluation interface
    # ------------------------------------------------------------------

    def evaluate(self, energy: np.ndarray) -> np.ndarray:
        """
        Evaluate f(E) at arbitrary energy values via the pre-built interpolator.

        Parameters
        ----------
        energy : array-like
            Relative binding energies at which to evaluate f(E).

        Returns
        -------
        f : ndarray  – same shape as *energy*; zero for unbound (E <= 0) states.
        """
        if self._f_interp is None:
            self.compute_distribution_function()
        e = np.asarray(energy)
        return self._f_interp(e)

    def evaluate_at_particles(self) -> np.ndarray:
        """
        Convenience wrapper: evaluate f(E_i) at every particle's binding energy.

        Returns
        -------
        f_particles : (N,) ndarray
        """
        e = self.compute_energies()
        return self.evaluate(e)

    # ------------------------------------------------------------------
    # Step 7 – sanity check: reconstruct density
    # ------------------------------------------------------------------

    def reconstruct_density(self, n_velocity: int = 64) -> tuple:
        """
        Validate f(E) by reconstructing the density profile via:

            rho(r) = 4 pi integral_0^{v_esc(r)} f(E(r,v)) v^2 dv

        where v_esc(r) = sqrt(2 * Psi(r)).

        Parameters
        ----------
        n_velocity : int
            Number of velocity quadrature points (default: 64).

        Returns
        -------
        psi_grid   : (n_grid,) ndarray
        rho_input  : (n_grid,) ndarray  – original density profile
        rho_recon  : (n_grid,) ndarray  – reconstructed density profile
        """
        if self._f_interp is None:
            self.compute_distribution_function()

        psi_grid, rho_grid = self.compute_density_profile()
        rho_recon = np.zeros(self.n_grid)

        for j, psi_j in enumerate(psi_grid):
            v_esc = np.sqrt(2.0 * psi_j)
            v = np.linspace(0.0, v_esc, n_velocity)
            E_j = psi_j - 0.5 * v**2
            f_v = self._f_interp(E_j)
            integrand = 4.0 * np.pi * f_v * v**2
            rho_recon[j] = np.trapz(integrand, v)

        return psi_grid, rho_grid, rho_recon

    # ------------------------------------------------------------------
    # Convenience: run all steps in sequence
    # ------------------------------------------------------------------

    def run(self) -> tuple:
        """
        Execute the full pipeline and return (psi_grid, f_grid).

        Equivalent to calling compute_distribution_function() directly.
        """
        return self.compute_distribution_function()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # (no further internal helpers)
    def plot_particle_values(self, ax=None, **kwargs):
        """
        Plot per-particle distribution function

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            plotting axis, by default None

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        f = self.evaluate_at_particles()
        """minf = np.min(f[f>0])
        f[f<=0] = minf / 10"""
        kwargs.setdefault("ls", "")
        kwargs.setdefault("marker", ".")
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.compute_energies(), f, ls="", marker=".")
        ylim = ax.get_ylim()
        # ax.axhspan(ymin=ylim[0], ymax=minf/5, fc="gray", alpha=0.3)
        ax.set_ylim(*ylim)
        return ax
