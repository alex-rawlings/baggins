import numpy as np
import scipy.integrate, scipy.constants, scipy.stats
from functools import cached_property
import os

from ..literature import *
from ..cosmology import *
from ..general import *
from ..mathematics import uniform_sample_sphere
from ..utils import read_parameters, write_parameters
from ..env_config import _logger

__all__ = []


MSOL = 1.998e30


class _GalaxyICBase:
    def __init__(self):
        """
        Basic properties that are always required for a GalaxyIC object.
        """
        pass

    def load_parameters(self, parameter_file):
        """
        Parameters
        ----------
        parameter_file : str, path-like
            path to parameter file describing the galaxy ICs

        Raises
        ------
        NotImplementedError
            if non-zero redshift set
        """
        self.parameter_file = parameter_file
        self.parameters = read_parameters(self.parameter_file)
        self.name = self.parameters.galaxyName
        self.save_location = os.path.join(self.parameters.saveLocation, self.name)
        self.figure_location = os.path.join(self.save_location,  self.parameters.figureLocation)
        self.data_location = os.path.join(self.save_location, self.parameters.dataLocation)
        self.lit_location = self.parameters.litDataLocation
        self.seed = self.parameters.randomSeed
        self._rng = np.random.default_rng(self.seed)
        self.simulation_time = self.parameters.simulationTime
        self.maximum_radius = self.parameters.maximumRadius
        self.minimum_radius = self.parameters.minimumRadius
        if self.simulation_time < 1e-6:
            #we are using redshift 0
            self.redshift = 0
        else:
            msg = "Only redshift 0 is currently supported"
            _logger.logger.error(msg)
            raise NotImplementedError(msg)
        # add redshift to the parameter file
        self.parameters.redshift = self.redshift
        self.cosmology = cosmology
        self.mass_units = "msol"
        #make output directories if not already existing
        os.makedirs(self.save_location, exist_ok=True)
        os.makedirs(self.figure_location, exist_ok=True)

    @property
    def maximum_radius(self):
        return self._maximum_radius

    @maximum_radius.setter
    def maximum_radius(self, val):
        self._maximum_radius = val
    
    @property
    def H0_in_Hz(self):
        # 3.24078e-20 : km/s/Mpc in Hz
        return 100 * self.cosmology["h"] * 3.24078e-20

    @cached_property
    def hubble_redshifted(self):
        omega_L = self.cosmology["omega_L"]
        omega_M = self.cosmology["omega_M"]
        return self.H0_in_Hz * np.sqrt(omega_L + omega_M*(1+self.redshift)**3) #in Hz

    @cached_property
    def critical_density(self):
        conversion = (1e3*scipy.constants.parsec)**3 / MSOL
        return 3 * self.hubble_redshifted**2 / (8 * np.pi * scipy.constants.G) * conversion


class _StellarComponent(_GalaxyICBase):
    def __init__(self, parameter_file=None):
        """
        Base class for the stellar component of a GalaxyIC object.

        Parameters
        ----------
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        if parameter_file is not None:
            self.load_parameters(parameter_file=parameter_file)
        self.total_mass = None
        try:
            self.anisotropy_radius = self.parameters.anisotropyRadius
        except AttributeError:
            self.anisotropy_radius = None
        self.particle_mass = self.parameters.stellarParticleMass
        self.softening = self.parameters.stellar_softening

    @property
    def log_total_mass(self):
        return np.log10(self.total_mass)


class _StellarCusp(_StellarComponent):
    def __init__(self, parameter_file=None):
        """
        Class describing a cuspy (Dehnen) stellar component.

        Parameters
        ----------
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        self.scale_radius = self.parameters.stellarScaleRadius
        self.gamma = self.parameters.stellarGamma
        self.total_mass = 10**self.parameters.logStellarMass


class _StellarCore(_StellarComponent):
    def __init__(self, parameter_file=None):
        """
        Class describing a cored (Terzic) stellar component

        Parameters
        ----------
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        self.distance_modulus = self.parameters.distanceModulus
        self.effective_radius = self.parameters.effectiveRadius
        self.sersic_index = self.parameters.sersicN
        self.transition_index = self.parameters.transitionIndex
        self.core_slope = self.parameters.coreSlope
        self.core_radius = self.parameters.coreRadius
        self.core_density = 10**self.parameters.logCoreDensity
        self.mass_light_ratio = self.parameters.M2Lratio
        self.stellar_distance_units = "arcsec"

    def to_kpc_units(self):
        """
        Convert input from units of arcsecs to kpc
        """
        assert(self.stellar_distance_units == "arcsec")
        self.core_radius = arcsec_to_kpc(self.distance_modulus, self.core_radius)
        self.effective_radius = arcsec_to_kpc(self.distance_modulus, self.effective_radius)
        self.stellar_distance_units = "kpc"
        # save the kpc values
        self.self.parameters.input_Re_in_kpc = self.effective_radius
        self.self.parameters.input_Rb_in_kpc = self.core_radius

    @cached_property
    def sersic_b_parameter(self):
        return sersic_b_param(self.sersic_index)

    @cached_property
    def total_mass(self):
        self.to_kpc_units()
        rhof = lambda r: r**2 * Terzic05(r, rhob=self.core_density*1e9, rb=self.core_radius, n=self.sersic_index, g=self.core_slope, b=self.sersic_b_parameter, Re=self.effective_radius, a=self.transition_index)
        return 4*np.pi * self.mass_light_ratio * scipy.integrate.quad(rhof, 1e-5, self.maximum_radius, limit=100)[0]


class _DMComponent(_GalaxyICBase):
    def __init__(self, stellar_mass, parameter_file=None):
        """
        Base class for the stellar component of a GalaxyIC object.

        Parameters
        ----------
        stellar_mass : float
            total stellar mass of the galaxy
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        if parameter_file is not None:
            self.load_parameters(parameter_file=parameter_file)
        self.particle_mass = self.parameters.DMParticleMass
        self.softening = self.parameters.DM_softening
        self.dm_scaling_relation = self.parameters.DM_mass_from.lower()
        self._stellar_mass = stellar_mass

    @property
    def peak_mass(self):
        return self._peak_mass

    @peak_mass.setter
    def peak_mass(self, val):
        try:
            if val is not None:
                self._peak_mass = val
            else:
                self._peak_mass = self.self.parameters.DM_peak_mass
                _logger.logger.info("DM Mass read from parameter file")
        except AttributeError:
            _logger.logger.info("Setting DM peak mass")
            if self.dm_scaling_relation == "moster":
                _logger.logger.info("Using Moster+10 DM scaling relation")
                self._peak_mass = 10**Moster10(self._stellar_mass, [1e10, 1e15], z=self.redshift, plotting=False)
            elif self.dm_scaling_relation == "girelli":
                _logger.logger.info("Using Girelli+20 DM scaling relation")
                self._peak_mass = 10**Girelli20(self._stellar_mass, [1e10, 1e15], z=self.redshift, plotting=False)
            elif self.dm_scaling_relation == "behroozi":
                _logger.logger.info("Using Behroozi+19 DM scaling relation")
                self._peak_mass = 10**Behroozi19(self._stellar_mass, [1e10, 1e15], z=self.redshift, plotting=False)
            else:
                msg = "Invalid scaling relation in parameter file!"
                _logger.logger.error(msg)
                raise RuntimeError(msg)
            self.self.parameters.DM_peak_mass = self._peak_mass

    @property
    def log_peak_mass(self):
        return np.log10(self.peak_mass)


class _DMHaloNFW(_DMComponent):
    def __init__(self, stellar_mass, parameter_file=None):
        """
        Class describing an NFW profile.

        Parameters
        ----------
        stellar_mass : float
            total stellar mass of the galaxy
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(stellar_mass=stellar_mass, parameter_file=parameter_file)
    
    @cached_property
    def overdensity(self):
        E2 = (self.star_info.general_info.hubble_redshifted / self.H0_in_Hz)**2
        # from Bryan and Norman 1998
        x = self.cosmology["omega_M"] * (1+self.redshift)**3 / E2
        return 18*np.pi**2 + 82*x - 39*x**2

    @property
    def concentration(self):
        #relation from Dutton+14
        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718*self.redshift**1.08)
        b = -0.097 + 0.024*self.redshift
        return 10**(a + b * np.log10(self.peak_mass / (1e2 / self.cosmology["h"])))

    @property
    def virial_radius(self):
        #in kpc
        return (self.virial_mass / (self.overdensity*4*np.pi/3 * self.critical_density))**(1/3)


class _DMHaloDehnen(_DMComponent):
    def __init__(self, stellar_mass, parameter_file=None):
        """
        Class describing a Dehnen DM halo.

        Parameters
        ----------
        stellar_mass : float
            total stellar mass of the galaxy
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(self, stellar_mass=stellar_mass, parameter_file=parameter_file)
        self.scale_radius = self.parameters.DMScaleRadius
        self.gamma = self.parameters.DMGamma

    @property
    def virial_radius(self):
        return self._virial_radius

    @virial_radius.setter
    def virial_radius(self, val):
        self._virial_radius = val


class _SMBH(_GalaxyICBase):
    def __init__(self, log_stellar_mass, parameter_file=None):
        """
        Class describing a SMBH component.

        Parameters
        ----------
        log_stellar_mass : float
            log10 of total stellar mass
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        if parameter_file is not None:
            self.load_parameters(parameter_file=parameter_file)
        self._log_stellar_mass = log_stellar_mass
        self.spin = self.parameters.BH_spin
    
    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, val):
        try:
            if val is not None:
                self._mass = val
            else:
                self._mass = self.parameters.BH_mass
                _logger.logger.info("BH Mass read from parameter file")
        except AttributeError:
            self._mass = 10**Sahu19(self._log_stellar_mass)
            self.parameters.BH_mass = self._mass
            _logger.logger.info("BH Mass determined from Sahu+19 relation")

    @property
    def log_mass(self):
        return np.log10(self.mass)

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, val):
        if isinstance(val, str):
            valid_str = True
            # choose spin parameters from the following distributions
            if self.parameters.BH_spin_from.lower() == "zlochower_dry":
                bh_spin_params = zlochower_dry_spins
            elif self.parameters.BH_spin_from.lower() == "zlochower_cold":
                bh_spin_params = zlochower_cold_spins
            elif self.parameters.BH_spin_from.lower() == "zlochower_hot":
                bh_spin_params = zlochower_hot_spins
            else:
                valid_str = False
                bh_spin_params = None
            # set up random spins
            if valid_str:
                spin_mag = scipy.stats.beta.rvs(*bh_spin_params, random_state=self._rng)
                t, p = uniform_sample_sphere(1, rng=self._rng)
                self._spin = spin_mag * np.array([
                                                np.sin(t) * np.cos(p),
                                                np.sin(t) * np.sin(p),
                                                np.cos(t)
                                                ]).flatten()

            else:
                _logger.logger.warning("Invalid spin distribution parameters given: setting spin to [0,0,0]")
                self._spin = np.array([0,0,0])
        else:
            assert len(val) == 3
            self._spin = val
        #save the new spin value
        self.parameters.BH_spin = self._spin

