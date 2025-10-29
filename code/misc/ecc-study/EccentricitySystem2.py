import itertools
import os.path
import argparse
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from math import erf
import pickle
from numba import njit


# helper functions
@njit
def erf_array(x):
    if isinstance(x, float):
        return erf(x)
    n = x.size
    out = np.empty_like(x)
    for i in range(n):
        out[i] = erf(x[i])
    return out

@njit
def euclid_norm(x):
    return np.sqrt(np.sum(x**2, axis=0))

@njit
def dynamical_friction(v, G, m1, m2, rho, logL, stellar_sigma):
    """
    Determine dynamical friction contribution

    Parameters
    ----------
    v : array-like
        velocity vector

    Returns
    -------
    : array-like
        dynamical friction contribution to acceleration
    """
    mbin = m1 + m2
    def _set_vel(m, v):
        v_single = v * m / mbin
        v_single_norm = euclid_norm(v_single)
        return v_single, v_single_norm

    def _df_taylor(m):
        v_single, v_single_norm = _set_vel(m, v)
        return -(v_single * 8*np.sqrt(np.pi) * m * rho * logL / (3*np.sqrt(2) * stellar_sigma**3))

    def _df_general(m):
        v_single, v_single_norm = _set_vel(m, v)
        X = v_single_norm/(np.sqrt(2)*stellar_sigma) 
        return -(4*np.pi * G**2 * m * rho * logL * (erf_array(X) - 2*X/np.sqrt(np.pi)*np.exp(-X**2)) * v_single/v_single_norm**3)

    v1_norm = _set_vel(m1, v)[1]
    v2_norm = _set_vel(m2, v)[1]
    df1 = _df_taylor(m1) if v1_norm/stellar_sigma < 1e-3 else _df_general(m1)
    df2 = _df_taylor(m2) if v2_norm/stellar_sigma < 1e-3 else _df_general(m2)
    return df1 + df2

@njit
def ellipsoid_accel(x, G, rho, Avec):
    """
    Potential well modelled by spheroidal potential with some ellipticity

    Parameters
    ----------
    x : array-like
        position coordinates

    Returns
    -------
    : array-like
        acceleration from ellipsoid potential
    """
    return -2. * np.pi * G * rho * Avec * x

@njit
def df_decoupling_factor(x, v, a, a_hard):
    """
    Decoupling factor as SMBH separation decreases

    Parameters
    ----------
    x : array-like
        relative distance vector between BHs
    v : array-like
        velocity vector

    Returns
    -------
    : array-like
        cut-off factor
    """
    #a[a<0] = np.inf
    cutoff_point = a_hard * 2
    cutoff_scale = a_hard * 0.5
    return 1/(1+np.exp(-(a-cutoff_point)/cutoff_scale))

@njit
def accel(x, v, G, m1, m2, rho, Avec, a, a_hard, stellar_sigma, logL):
    """
    Determine acceleration

    Parameters
    ----------
    x : array-like
        relative distance vector between BHs
    v : array-like
        velocity vector

    Returns
    -------
    : array-like
        acceleration vector
    """
    mbin = m1 + m2
    r = euclid_norm(x)
    return (-G * mbin/r**3 * x 
            + ellipsoid_accel(x=x * m1/mbin, G=G, rho=rho, Avec=Avec)
            + ellipsoid_accel(x=x * m2/mbin, G=G, rho=rho, Avec=Avec)
            + dynamical_friction(v=v, G=G, m1=m1, m2=m2, rho=rho, logL=logL, stellar_sigma=stellar_sigma)*df_decoupling_factor(x,v, a, a_hard)
            )

@njit
def integrate_dydt(t, y, G, m1, m2, rho, logL, stellar_sigma, a, a_hard, Avec):
    dvdt = accel(y[:2], y[2:], G=G, m1=m1, m2=m2, rho=rho, logL=logL, stellar_sigma=stellar_sigma, a=a, a_hard=a_hard, Avec=Avec)
    dy = np.full_like(y, np.nan)
    dy[:2] = y[2:]
    dy[2:] = dvdt
    return dy

class EccentricitySystem:
    def __init__(self, m1, m2, ellipticity=None, rho=None, stellar_sigma=None, rmax=None):
        """
        Base class for representing a system. Parameter 'm1' is automatically set to the more massive BH.

        Parameters
        ----------
        m1 : float
            BH mass 1
        m2 : float
            BH mass 2
        ellipticity : float, optional
            system ellipticity, by default None
        rho : float, optional
            stellar 3D density, by default None
        stellar_sigma : float, optional
            stellar velocity dispersion, by default None
        rmax : float, optional
            maximum radius of the system, by default None
        """
        # set units Gyr, 1e10 Msol, kpc
        self._G = 44900
        self._km_per_s = 1.02201216 # kpc/Gyr

        self.m1 = max(m1, m2)
        self.m2 = min(m1, m2)
        self.ellipticity = ellipticity
        self.rho = rho
        self.stellar_sigma = stellar_sigma * self._km_per_s
        self.rmax = rmax

        # pre-instantiate variables that result from integration
        self._bs = None
        self._v0s = None
        self._theta = None
        self._ecc = None

    @property
    def ellipticity(self):
        return self._ellipticity

    @ellipticity.setter
    def ellipticity(self, v):
        if v is not None and (v < 0 or v >= 1):
            raise AssertionError("Ellipticity must be between 0 and 1!")
        self._ellipticity = v

    @property
    def mbin(self):
        return self.m1 + self.m2

    @property
    def reduced_mass(self):
        return self.m1 * self.m2 / self.mbin

    @property
    def ecc(self):
        return self._ecc

    @property
    def bs(self):
        return self._bs

    @property
    def theta(self):
        return self._theta

    @property
    def v0s(self):
        return self._v0s

    @property
    def argument_of_periapsis(self):
        r = np.linalg.norm(self.x, axis=0)
        rdot = np.sum(self.x*self.v,axis=0)/r
        i = np.nonzero(rdot>0)[0][0]
        h = np.cross(self.x[:,i], self.v[:,i])
        evec = np.cross(self.v[:,i], [0,0,h])[:2]/(self._G*self.mbin) - self.x[:,i]/r[i]
        return np.arctan2(evec[1],evec[0])

    @property
    def b90(self):
        return 2 * self._G * self.m1 / self.stellar_sigma**2

    @property
    def logL(self):
        return np.log(self.rmax / self.b90)

    @property
    def a_hard(self):
        return self._G * self.reduced_mass / (4 * self.stellar_sigma**2)

    @property
    def r_infl(self):
        return np.cbrt(self.mbin/(self.rho*(4/3*np.pi)))

    @property
    def Avec(self):
        e2s = self.ellipticity**2
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*self.ellipticity)*np.log((1+self.ellipticity)/(1-self.ellipticity)))
        A3 = 2*(1-e2s)/e2s*(1/(2*self.ellipticity)*np.log((1+self.ellipticity)/(1-self.ellipticity)) - 1)
        return np.array([[A3],[A1]])

    @classmethod
    def load(cls, fname):
        """
        Load a previously-run class (useful for when curves have been calculated).

        Parameters
        ----------
        fname : str
            file name to load

        Returns
        -------
        C : EccentricitySystem
            class instance
        """
        with open(fname, "rb") as f:
            C = pickle.load(f)
        return C

    def calculate_deflection_angle(self, x, v):
        """
        Determine the deflection angle.

        Parameters
        ----------
        x : array-like
            position coordinates
        v : array-like
            velocity coordinates

        Returns
        -------
        : array-like
            deflection angles
        """
        r = np.linalg.norm(x, axis=0)
        rdot = np.sum(x*v, axis=0)/r
        i = np.nonzero(rdot>0)[0][0]
        E = self.orbital_energy(x[:,i], v[:,i])
        L = np.cross(x[:,i], v[:,i])
        return 2*np.arctan(self._G * self.mbin/(L*np.sqrt(2*E)))

    def _integrate(self, x0, v0, ts):
        """
        Integrate system acceleration

        Parameters
        ----------
        x0 : array-like
            initial separation
        v0 : array-like
            initial velocity
        ts : array-like
            times for integration sampling

        Returns
        -------
        t : array-like
            times of sampled integrals
        x : array-like
            separations
        v : array-like
            velocities
        """
        def dydt_wrapper(t,y):
            return integrate_dydt(t, y, G=self._G, m1=self.m1, m2=self.m2, rho=self.rho, logL=self.logL, stellar_sigma=self.stellar_sigma, a=self.semimajor_axis(y[:2], y[2:]), a_hard=self.a_hard, Avec=self.Avec)

        def min_E(t,y):
            E = self.orbital_energy(y[:2], y[2:])
            return E + self._G*self.mbin/(2*self.a_hard)
        min_E.terminal=True

        res = solve_ivp(dydt_wrapper, (ts[0], ts[-1]), np.append(x0,v0),
                        t_eval=ts, #vectorized=True,
                        method='DOP853',
                        events=[min_E],
                        rtol=1e-5, atol=1e-8)
        t = res.t
        x = res.y[:2]
        v = res.y[2:]
        return t, x, v

    def orbital_energy(self, x, v):
        """
        Determine orbital energy of system.

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            energy
        """
        r = np.linalg.norm(x,axis=0)
        v2 = np.sum(v*v,axis=0)
        return .5 * v2 - self._G * self.mbin/r

    def eccentricity(self, x, v):
        """
        Determine eccentricity of binary.

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            eccentricities
        """
        # h = L / mu
        h2 = np.cross(x,v, axis=0)**2
        E = self.orbital_energy(x,v)
        return np.sqrt(1 + 2 * h2 * E / (self._G * self.mbin)**2)
    
    def semimajor_axis(self, x, v):
        """
        Determine semimajor axis of binary.

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            semimajor axes
        """
        return -self._G*self.mbin/(2*self.orbital_energy(x,v))

    def integrate(self, args):
        """
        Helper for multiprocessing of integration

        Parameters
        ----------
        args : tuple
            arguments for _integrate() function, must be (b, v, r) where b is the impact parameter, v is the initial velocity, and r is the initial separation

        Returns
        -------
        t : array-like
            times of sampled integrals
        x : array-like
            separations
        v : array-like
            velocities
        """
        b, v, r = args
        x0 = [r,b]
        v0 = [-v,0]
        tmax = 150 * r/v
        ts = np.concatenate(([0], np.linspace(0.5*r/v,1.5*r/v, 50), np.linspace(1.5*r/v,tmax, 500)[1:]))
        return self._integrate(x0, v0, ts)

    def calculate_b_theta_e_curves(self, v0s, bs, r0=None):
        """
        Calculate the impact parameter - eccentricity curves. Calculation results are saved to class members 'theta' and 'ecc'.

        Parameters
        ----------
        v0s : array-like
            initial velocities
        bs : array-like
            impact parameters
        r0 : array-like, optional
            initial separations, by default None
        """
        self._bs = bs
        self._v0s = v0s * self._km_per_s
        if r0 is None:
            r0 = self.r_infl
        Bs, V0s = np.meshgrid(self.bs, self.v0s)
        n_tot = np.prod(Bs.shape)
        efin = [None] * n_tot
        deflection_angle = [None] * n_tot
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.integrate, x): i for i, x in enumerate(zip(Bs.ravel(), V0s.ravel(), itertools.repeat(r0)))}
            for future in tqdm(as_completed(futures), total=n_tot, desc="Calculating curves", unit="task"):
                idx = futures[future]
                t, x, v = future.result()
                efin[idx] = self.eccentricity(x[:,-1], v[:,-1])
                deflection_angle[idx] = self.calculate_deflection_angle(x, v)
        
        '''with multiprocessing.Pool() as pool:
            for i, (t, x, v) in enumerate(tqdm(
                pool.imap(self.integrate, zip(
                    Bs.ravel(),
                    V0s.ravel(),
                    itertools.repeat(r0)
                )),
                total=n_tot,
                desc="Calculating curves"
            )):
                efin.append(self.eccentricity(x[:,-1], v[:,-1]))
                deflection_angle.append(self.calculate_deflection_angle(x, v))
        '''

        self._theta = np.array(deflection_angle).reshape(V0s.shape)
        self._ecc = np.array(efin).reshape(Bs.shape)


    def save(self, fname, exist_ok=True):
        """
        Save a class instance to a .pickle file.

        Parameters
        ----------
        fname : str, path-like
            file name to save to
        exist_ok : bool, optional
            allow overwriting of previous files, by default True

        Raises
        ------
        RuntimeError
            if a similarly-named file exists and 'exist_ok' is False
        """
        if os.path.exists(fname) and not exist_ok:
            raise RuntimeError(f"File {fname} exists!")
        with open(fname, "wb") as f:
            pickle.dump(self, f, protocol=-1)
        print(f"Saved {fname}")

    def plot_theta_ecc_curve(self, v0, ax=None, shift=0):
        """
        Plot the calculated curves. If v0 is a velocity value not directly sampled, the closest-sampled initial velocity will be used.

        Parameters
        ----------
        v0 : float
            initial velocity
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        shift : float, optional
            deflection angle shift, by default 0

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$e_\mathrm{h}$")
        i = np.argmin(abs(np.array(self.v0s)-v0))
        ax.plot(np.degrees(self.theta[i]) + shift, self.ecc[i], label=f"$v_0 = {self.v0s[i]:3.0f}"r'/\mathrm{km\,s^{-1}}$')
        ax.legend()
        return ax

    def plot_theta_ecc_curve_all_velocities(self, ax=None, shift=0):
        """
        Plot the calculated curves for all sampled velocities.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            plotting axes, by default None
        shift : float, optional
            deflection angle shift, by default 0

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$e_\mathrm{h}$")
        lines = LineCollection(list(map(np.column_stack, zip(np.degrees(self.theta)+shift, self.ecc))), array=self.v0s, cmap="cividis", lw=2)
        ax.add_collection(lines)
        ax.set_xlim(0,180)
        cbar = fig.colorbar(lines)
        cbar.set_label("$v_0/\mathrm{km\,s^{-1}}$")
        return ax

    def add_artifical_end_to_curve(self, t_shift=0):
        """
        Add an artifical end to the curve for plotting purposes.

        Parameters
        ----------
        t_shift : float, optional
            deflection angle shift, by default 0
        """
        self._theta = np.insert(self._theta, 0, np.pi-t_shift)
        self._ecc = np.append(self._ecc, 1)
        self._bs = np.insert(self._bs, 0, 0)

    def __str__(self):
        s = "EccentricitySystem object with:\n"
        params = ["m1", "m2", "ellipticity", "rho", "stellar_sigma", "rmax", "r_infl", "logL"]
        for p in params:
            try:
                s2 = f" {p}: {getattr(self, p):.2e}\n"
            except TypeError:
                s2 = f" {p}: {getattr(self, p)}\n"
            s += s2
        return s


class  EccentricitySystemScanner:
    def __init__(self, ecc_sys, ellipticity, rmax, v0s, bs, r0=None, data_path=""):
        """
        Helper class for scanning different parameter combinations for the EccentricitySystem

        Parameters
        ----------
        ecc_sys : EccentricitySystem
            base class to scan
        ellipticity : float, optional
            system ellipticity, by default None
        rmax : float, optional
            maximum radius of the system, by default None
        v0s : array-like
            initial velocities
        bs : array-like
            impact parameters
        r0 : array-like, optional
            initial separations, by default None
        data_path : str, optional
            path to save data to, by default ""

        Raises
        ------
        TypeError
            for incorrect inputs of 'ellipticity' and 'rmax'
        """
        self.ecc_sys = ecc_sys
        self.v0s = v0s
        self.bs = bs
        self.r0 = r0
        self.data_path = data_path
        if isinstance(ellipticity, (list, tuple)):
            self.ellipticity = ellipticity
        elif isinstance(ellipticity, (float, int)):
            self.ellipticity = [ellipticity]
        else:
            raise TypeError(f"Unrecognised input {type(ellipticity)} for parameter 'ellipticity'!")
        if isinstance(rmax, (list, tuple)):
            self.rmax = rmax
        elif isinstance(rmax, (float, int)):
            self.rmax = [rmax]
        else:
            raise TypeError(f"Unrecognised input {type(rmax)} for parameter 'rmax'!")

    def save(self, exist_ok=True):
        """
        Save an instance of the system.

        Parameters
        ----------
        exist_ok : bool, optional
            allow overwriting of similarly-named files, by default True
        """
        fname = os.path.join(self.data_path, f"scan2_ellip{self.ecc_sys.ellipticity:.3f}_rmax{self.ecc_sys.rmax:.2f}.pickle")
        self.ecc_sys.save(fname, exist_ok=exist_ok)

    def scan(self, exist_ok=True):
        """
        Method to scan parameter space.

        Parameters
        ----------
        exist_ok : bool, optional
            allow overwriting of similarly-named files, by default True
        """
        for ellipticity, rmax in itertools.product(self.ellipticity, self.rmax):
            self.ecc_sys.rmax = rmax
            self.ecc_sys.ellipticity = ellipticity
            print(self.ecc_sys)
            self.ecc_sys.calculate_b_theta_e_curves(self.v0s, self.bs, self.r0)
            self.save(exist_ok=exist_ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make model curves for deflection angle - eccentricity relation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--load", type=str, dest="datafile", help="load previous run", default=None)
    parser.add_argument("--velocity", type=float, dest="vel", help="initial velocity", default=450)
    args = parser.parse_args()

    # warm up numba
    _ = integrate_dydt(t=0, y=np.ones(4, dtype=float), G=1, m1=1, m2=1, rho=1, logL=1, stellar_sigma=1, a=np.array([12]), a_hard=1, Avec=np.array([1,1]))

    if args.datafile is None:
        # set up the base system here
        ecc_sys = EccentricitySystem(m1=1e-2, m2=1e-2, rho=40, stellar_sigma=200)
        scanner = EccentricitySystemScanner(
            ecc_sys=ecc_sys,
            ellipticity=[0.85], # these are the values we scan
            rmax = [10, 20], # these are the values we scan
            v0s = np.linspace(350, 600, 10),
            bs = np.linspace(1, 25, 300) * 1e-3,  # in pc
            data_path="data"
        )
        scanner.scan()
    else:
        # load a previous result
        ecc_sys = EccentricitySystem.load(args.datafile)
        print(ecc_sys)
        ax = ecc_sys.plot_theta_ecc_curve_all_velocities()
        plt.savefig("scan_res.png")
