import itertools
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf


class EccentricitySystem:
    def __init__(self, m1, m2, e_spheroid, rho, stellar_sigma, system_rmax):
        # set units Gyr, 1e10 Msol, kpc
        self._G = 44900
        self._km_per_s = 1.02201216 # kpc/Gyr

        self.m1 = max(m1, m2)
        self.m2 = min(m1, m2)
        self.e_spheroid = e_spheroid
        self.rho = rho
        self.stellar_sigma = stellar_sigma * self._km_per_s
        self.system_rmax = system_rmax

        # set the system
        e2s = self.e_spheroid**2
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*self.e_spheroid)*np.log((1+self.e_spheroid)/(1-self.e_spheroid)))
        A3 = 2*(1-e2s)/e2s*(1/(2*self.e_spheroid)*np.log((1+self.e_spheroid)/(1-self.e_spheroid)) - 1)
        self.Avec = np.array([[A3],[A1]])
        self.rho = rho
        self.b90 = 2 * self._G * self.m1 / self.stellar_sigma**2
        self.logL = np.log(self.system_rmax / self.b90)
        self.a_hard = self._G * self.reduced_mass / (4 * self.stellar_sigma**2)
        self.r_infl = np.cbrt(self.mbin/(self.rho*(4/3*np.pi)))

    @property
    def mbin(self):
        return self.m1 + self.m2

    @property
    def reduced_mass(self):
        return self.m1 * self.m2 / self.reduced_mass

    def ellipsoid_accel(self, x):
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
        return -2 * np.pi * self._G * self.rho * self.Avec * x

    def dynamical_friction(self, x, v):
        ...