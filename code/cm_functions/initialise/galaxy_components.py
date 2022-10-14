import numpy as np
import scipy.integrate, scipy.constants, scipy.stats
from functools import cached_property
import os

from ..literature import *
from ..cosmology import *
from ..general import *
from ..mathematics import uniform_sample_sphere
from ..utils import read_parameters
from ..env_config import _cmlogger

__all__ = []

_logger = _cmlogger.copy(__file__)

MSOL = 1.998e30


class _GalaxyICBase:
    def __init__(self, parameter_file):
        """
        Basic properties that are always required for a GalaxyIC object.
        
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
        self.name = self.parameters["general"]["galaxy_name"]
        self.save_location = os.path.join(self.parameters["file_locations"]["save_location"], self.name)
        self.figure_location = os.path.join(self.save_location,  self.parameters["file_locations"]["figure_location"])
        self.data_location = os.path.join(self.save_location, self.parameters["file_locations"]["data_location"])
        self.seed = self.parameters["general"]["random_seed"]
        self._rng = np.random.default_rng(self.seed)
        self.simulation_start_time = self.parameters["general"]["simulation_start_time"]["value"]
        self.maximum_radius = self.parameters["general"]["maximum_radius"]["value"]
        self.minimum_radius = self.parameters["general"]["minimum_radius"]["value"]
        try:
            assert self.simulation_start_time < 0
            #we are using redshift 0
            self.redshift = 0
        except AssertionError:
            _logger.logger.exception("Only redshift 0 is currently supported", exc_info=True)
            raise
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
    def __init__(self, parameter_file):
        """
        Base class for the stellar component of a GalaxyIC object.

        Parameters
        ----------
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        self.total_mass = None
        if self.parameters["stars"]["anisotropy_radius"]["value"] is None:
            self.anisotropy_radius = None
        else:
            self.anisotropy_radius = self.parameters["stars"]["anisotropy_radius"]["value"]
        self.particle_mass = self.parameters["stars"]["particle_mass"]["value"]
        self.softening = self.parameters["stars"]["softening"]["value"]

    @property
    def log_total_mass(self):
        return np.log10(self.total_mass)


class _StellarCusp(_StellarComponent):
    def __init__(self, parameter_file):
        """
        Class describing a cuspy (Dehnen) stellar component.

        Parameters
        ----------
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        pars = self.parameters["stars"]["cuspy"]
        self.scale_radius = pars["scale_radius"]["value"]
        self.gamma = pars["gamma"]
        self.total_mass = 10**pars["log_total_mass"]


class _StellarCore(_StellarComponent):
    def __init__(self, parameter_file):
        """
        Class describing a cored (Terzic) stellar component

        Parameters
        ----------
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        pars = self.parameters["stars"]["cored"]
        self.distance_modulus = pars["distance_modulus"]
        self.effective_radius = pars["effective_radius"]["value"]
        self.sersic_index = pars["sersic_n"]
        self.transition_index = pars["transition_index"]
        self.core_slope = pars["core_slope"]
        self.core_radius = pars["core_radius"]["value"]
        self.core_density = 10**pars["log_core_density"]["value"]
        self.mass_light_ratio = pars["mass_light_ratio"]
        self.stellar_distance_units = "arcsec"
        # TODO checks for unit consistency
        self.to_kpc_units()
        rhof = lambda r: r**2 * Terzic05(r, rhob=self.core_density*1e9, rb=self.core_radius, n=self.sersic_index, g=self.core_slope, b=self.sersic_b_parameter, Re=self.effective_radius, a=self.transition_index)
        self.total_mass = 4*np.pi * self.mass_light_ratio * scipy.integrate.quad(rhof, 1e-5, self.maximum_radius, limit=100)[0]

    def to_kpc_units(self):
        """
        Convert input from units of arcsecs to kpc
        """
        assert(self.stellar_distance_units == "arcsec")
        self.core_radius = arcsec_to_kpc(self.distance_modulus, self.core_radius)
        self.effective_radius = arcsec_to_kpc(self.distance_modulus, self.effective_radius)
        self.stellar_distance_units = "kpc"
    

    @cached_property
    def sersic_b_parameter(self):
        return sersic_b_param(self.sersic_index)


class _DMComponent(_GalaxyICBase):
    def __init__(self, stellar_mass, parameter_file):
        """
        Base class for the stellar component of a GalaxyIC object.

        Parameters
        ----------
        stellar_mass : float
            total stellar mass of the galaxy
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        self.particle_mass = self.parameters["dm"]["particle_mass"]["value"]
        self.softening = self.parameters["dm"]["softening"]["value"]
        self.dm_scaling_relation = self.parameters["dm"]["mass_relation"].lower()
        self._stellar_mass = stellar_mass
        self.peak_mass = None

    @property
    def peak_mass(self):
        return self._peak_mass

    @peak_mass.setter
    def peak_mass(self, val):
        try:
            if val is not None:
                self._peak_mass = val
            else:
                self._peak_mass = self.parameters["calculated"]["dm"]["peak_mass"]
                _logger.logger.info("DM Mass read from parameter file")
        except KeyError:
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

    @property
    def log_peak_mass(self):
        return np.log10(self.peak_mass)


class _DMHaloNFW(_DMComponent):
    def __init__(self, stellar_mass, parameter_file):
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
    def __init__(self, stellar_mass, parameter_file):
        """
        Class describing a Dehnen DM halo.

        Parameters
        ----------
        stellar_mass : float
            total stellar mass of the galaxy
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(stellar_mass=stellar_mass, parameter_file=parameter_file)
        self.scale_radius = self.parameters["dm"]["Dehnen"]["scale_radius"]["value"]
        self.gamma = self.parameters["dm"]["Dehnen"]["gamma"]

    @property
    def virial_radius(self):
        return self._virial_radius

    @virial_radius.setter
    def virial_radius(self, val):
        self._virial_radius = val


class _SMBH(_GalaxyICBase):
    def __init__(self, log_stellar_mass, parameter_file):
        """
        Class describing a SMBH component.

        Parameters
        ----------
        log_stellar_mass : float
            log10 of total stellar mass
        parameter_file : str, path-like, optional
            path to parameter file describing the galaxy ICs, by default None
        """
        super().__init__(parameter_file=parameter_file)
        self._log_stellar_mass = log_stellar_mass
        self.mass = None
        self.spin_relation = self.parameters["bh"]["spin_relation"].lower()
        try:
            self.spin = self.parameters["calculated"]["bh"]["spin"]
        except KeyError:
            self.spin = self.parameters["bh"]["set_spin"]
        self.softening = self.parameters["bh"]["softening"]["value"]
    
    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, val):
        try:
            if val is not None:
                self._mass = val
            else:
                self._mass = self.parameters["calculated"]["bh"]["mass"]
                _logger.logger.info("BH Mass read from parameter file")
        except KeyError:
            self._mass = 10**Sahu19(self._log_stellar_mass)
            _logger.logger.info("BH Mass determined from Sahu+19 relation")

    @property
    def log_mass(self):
        return np.log10(self.mass)

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, val):
        if isinstance(val, str) or val is None:
            valid_str = True
            # choose spin parameters from the following distributions
            if self.spin_relation == "zlochower_dry":
                bh_spin_params = zlochower_dry_spins
            elif self.spin_relation == "zlochower_cold":
                bh_spin_params = zlochower_cold_spins
            elif self.spin_relation == "zlochower_hot":
                bh_spin_params = zlochower_hot_spins
            else:
                valid_str = False
                bh_spin_params = None
            # set up random spins
            if valid_str:
                _logger.logger.info(f"Generating BH spins from {self.spin_relation}")
                spin_mag = scipy.stats.beta.rvs(*bh_spin_params.values(), random_state=self._rng)
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

