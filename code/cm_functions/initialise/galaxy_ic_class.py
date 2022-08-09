import copy
import numpy as np
import scipy.integrate, scipy.constants
from functools import cached_property
import os

from ..literature import *
from ..cosmology import *
from ..general import *
from ..env_config import _logger

__all__ = ['galaxy_ic_base', 'ic_general_data', 'stellar_component', 'stellar_cuspy_ic', 'stellar_cored_ic', 'dm_component', 'dm_halo_NFW', 'dm_halo_dehnen', 'smbh']

class galaxy_ic_base:
    def __init__(self, parameter_file, stars=False, dm=False, bh=False):
        self.general = ic_general_data(parameter_file)
        if stars:
            if parameter_file.stellarCored:
                self.stars = stellar_cored_ic(parameter_file, self.general)
            else:
                self.stars = stellar_cuspy_ic(parameter_file, self.general)
        else:
            self.stars = None
        if dm:
            if parameter_file.use_NFW:
                self.dm = dm_halo_NFW(parameter_file, self.stars)
            else:
                self.dm = dm_halo_dehnen(parameter_file, self.stars)
        else:
            self.dm = None
        if bh:
            self.bh = smbh(parameter_file, self.stars)
        else:
            self.bh = None

    def to_gadget_mass_units(self):
        _logger.logger.info('Converting mass units to gadget')
        assert(self.general.mass_units == 'msol')
        self.stars.particle_mass /= 1e10
        self.stars.total_mass /= 1e10
        self.dm.particle_mass /= 1e10
        self.dm.peak_mass = (False, self.dm.peak_mass/1e10)
        self.bh.mass = (False, self.bh.mass/1e10)
        self.general.mass_units = 'gadget'

    def print_masses(self):
        print('Printing masses')
        print('Mass unit: {:s}'.format(self.general.mass_units))
        print('Stellar particle mass: {:.3e}'.format(self.stars.particle_mass))
        print('Stellar total mass: {:.3e}'.format(self.stars.total_mass))
        print('DM particle mass: {:.3e}'.format(self.dm.particle_mass))
        print('DM total mass: {:.3e}'.format(self.dm.peak_mass))
        print('BH mass: {:.3e}'.format(self.bh.mass))


class ic_general_data:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.name = parameter_file.galaxyName
        self.save_location = parameter_file.saveLocation + '/' + self.name
        self.figure_location = self.save_location + '/' + parameter_file.figureLocation
        self.data_location = self.save_location + '/' + parameter_file.dataLocation
        self.lit_location = parameter_file.litDataLocation
        self.seed = parameter_file.randomSeed
        self.simulation_time = parameter_file.simulationTime
        self._maximum_radius = parameter_file.maximumRadius
        self.minimum_radius = parameter_file.minimumRadius
        try:
            self.anisotropy_radius = parameter_file.anisotropyRadius
        except AttributeError:
            self.anisotropy_radius = None
        if self.simulation_time < 1e-6:
            #we are using redshift 0
            self.redshift = 0
        else:
            self.redshift = time2z(self.simulation_time, pres=True)
        self.parameter_file.redshift = self.redshift
        self.cosmology = cosmology
        self.mass_units = 'msol'
        #make output directories if not already existing
        if(not os.path.isdir(self.save_location)):
            os.mkdir(self.save_location)
        if(not os.path.isdir(self.figure_location)):
            os.mkdir(self.figure_location)

    @property
    def maximum_radius(self):
        return self._maximum_radius

    @maximum_radius.setter
    def maximum_radius(self, val):
        self._maximum_radius = val

    @cached_property
    def hubble_redshifted(self):
        H0 = 100 * self.cosmology['h'] #in km/s/Mpc
        conversion = 3.24078e-20 #km/s/Mpc in Hz
        omega_L = self.cosmology['omega_L']
        omega_M = self.cosmology['omega_M']
        return H0 * np.sqrt(omega_L + omega_M*(1+self.redshift)**3) * conversion #in Hz

    @cached_property
    def critical_density(self):
        conversion = (1e3*scipy.constants.parsec)**3 / (1.988e30)
        return 3 * self.hubble_redshifted**2 / (8 * np.pi * scipy.constants.G) * conversion


class stellar_component:
    def __init__(self, parameter_file, general_info):
        self.parameter_file = parameter_file
        self.general_info = general_info
        self.particle_mass = parameter_file.stellarParticleMass
        self.softening = parameter_file.stellar_softening

    @property
    def log_total_mass(self):
        return np.log10(self.total_mass)


class stellar_cuspy_ic(stellar_component):
    def __init__(self, parameter_file, general_info):
        stellar_component.__init__(self, parameter_file, general_info)
        self.scale_radius = parameter_file.stellarScaleRadius
        self.gamma = parameter_file.stellarGamma
        self.total_mass = 10**parameter_file.logStellarMass


class stellar_cored_ic(stellar_component):
    def __init__(self, parameter_file, general_info):
        stellar_component.__init__(self, parameter_file, general_info)
        self.distance_modulus = parameter_file.distanceModulus
        self.effective_radius = parameter_file.effectiveRadius
        self.sersic_index = parameter_file.sersicN
        self.transition_index = parameter_file.transitionIndex
        self.core_slope = parameter_file.coreSlope
        self.core_radius = parameter_file.coreRadius
        self.core_density = 10**parameter_file.logCoreDensity
        self.mass_light_ratio = parameter_file.M2Lratio
        self.stellar_distance_units = 'arcsec'

    def to_kpc_units(self):
        assert(self.stellar_distance_units == 'arcsec')
        self.core_radius = arcsec_to_kpc(self.distance_modulus, self.core_radius)
        self.effective_radius = arcsec_to_kpc(self.distance_modulus, self.effective_radius)
        self.stellar_distance_units = 'kpc'
        #save the kpc values
        self.parameter_file.input_Re_in_kpc = self.effective_radius
        self.parameter_file.input_Rb_in_kpc = self.core_radius

    @cached_property
    def sersic_b_parameter(self):
        return sersic_b_param(self.sersic_index)

    @cached_property
    def total_mass(self):
        self.to_kpc_units()
        rhof = lambda r: r**2 * Terzic05(r, rhob=self.core_density*1e9, rb=self.core_radius, n=self.sersic_index, g=self.core_slope, b=self.sersic_b_parameter, Re=self.effective_radius, a=self.transition_index)
        return 4*np.pi * self.mass_light_ratio * scipy.integrate.quad(rhof, 1e-5, self.general_info.maximum_radius, limit=100)[0]


class dm_component:
    def __init__(self, parameter_file, star_info):
        self.parameter_file = parameter_file
        self.particle_mass = parameter_file.DMParticleMass
        self.softening = parameter_file.DM_softening
        self.dm_scaling_relation = parameter_file.DM_mass_from.lower()
        self.star_info = star_info

    @property
    def peak_mass(self):
        return self._peak_mass

    @peak_mass.setter
    def peak_mass(self, input):
        # TODO remove verbose inputs!
        verbose, value = input
        try:
            if value is not None:
                dm_mass = value
            else:
                dm_mass = self.parameter_file.DM_peak_mass
                _logger.logger.info('DM Mass read from parameter file')
            self._peak_mass = dm_mass
        except AttributeError:
            if self.dm_scaling_relation == 'moster':
                _logger.logger.info('Using Moster+10 DM scaling relation')
                self._peak_mass = 10**Moster10(self.star_info.total_mass, [1e10, 1e15], z=self.star_info.general_info.redshift, plotting=False)
                self.parameter_file.DM_peak_mass = self._peak_mass
            elif self.dm_scaling_relation == 'girelli':
                _logger.logger.info('Using Girelli+20 DM scaling relation')
                self._peak_mass = 10**Girelli20(self.star_info.total_mass, [1e10, 1e15], z=self.star_info.general_info.redshift, plotting=False)
                self.parameter_file.DM_peak_mass = self._peak_mass
            elif self.dm_scaling_relation == 'behroozi':
                _logger.logger.info('Using Behroozi+19 DM scaling relation')
                self._peak_mass = 10**Behroozi19(self.star_info.total_mass, [1e10, 1e15], z=self.star_info.general_info.redshift, plotting=False)
                self.parameter_file.DM_peak_mass = self._peak_mass
            else:
                raise RuntimeError('Invalid scaling relation in parameter file!')

    @property
    def log_peak_mass(self):
        return np.log10(self.peak_mass)


class dm_halo_NFW(dm_component):
    def __init__(self, parameter_file, star_info):
        dm_component.__init__(self, parameter_file, star_info)

    @cached_property
    def overdensity(self):
        z = self.star_info.general_info.redshift
        H0 = 100 * self.star_info.general_info.cosmology['h'] *  3.24078e-20 #in Hz
        E2 = (self.star_info.general_info.hubble_redshifted / H0)**2
        x = self.star_info.general_info.cosmology['omega_M']*(1+z)**3 / E2
        #from Bryan and Norman 1998
        return 18*np.pi**2 + 82*x - 39*x**2

    @property
    def concentration(self):
        #relation from Dutton+14
        z = self.star_info.general_info.redshift
        h = self.star_info.general_info.cosmology['h']
        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718*z**1.08)
        b = -0.097 + 0.024*z
        return 10**(a + b * np.log10(self.peak_mass / (1e2 / h)))

    @property
    def virial_radius(self):
        #in kpc
        return (self.virial_mass / (self.overdensity*4*np.pi/3 * self.star_info.general_info.critical_density))**(1/3)


class dm_halo_dehnen(dm_component):
    def __init__(self, parameter_file, star_info):
        dm_component.__init__(self, parameter_file, star_info)
        self.scale_radius = parameter_file.DMScaleRadius
        self.gamma = parameter_file.DMGamma

    @property
    def virial_radius(self):
        return self._virial_radius

    @virial_radius.setter
    def virial_radius(self, val):
        self._virial_radius = val


class smbh:
    def __init__(self, parameter_file, star_info):
        self.parameter_file = parameter_file
        self.star_info = star_info
        self.softening = parameter_file.BH_softening
        self._spin = parameter_file.BH_spin
        if isinstance(self._spin, str):
            self._spin = np.array([0,0,0])
        self.softening = parameter_file.BH_softening

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, input):
        verbose, value = input
        try:
            if value is not None:
                self._mass = value
            else:
                self._mass = self.parameter_file.BH_mass
                if verbose:
                    print('BH Mass read from parameter file')
        except AttributeError:
            self._mass = 10**Sahu19(self.star_info.log_total_mass)
            self.parameter_file.BH_mass = self._mass
            if verbose:
                print('BH Mass determined from Sahu+19 relation')

    @property
    def log_mass(self):
        return np.log10(self.mass)

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, val):
        self._spin = val
        #save the new spin value
        self.parameter_file.BH_spin = self._spin
