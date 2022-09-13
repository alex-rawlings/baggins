import numpy as np
from ...env_config import _logger
from . import HDF5Base

__all__ = ["HMQuantitiesData"]


class HMQuantitiesData(HDF5Base):
    def __init__(self) -> None:
        """
        Base class representing the quantities of a hierarchical model. Allows 
        for easy saving and loading of the class properties to and from a HDF5 
        file.
        """
        super().__init__()
    
    ##--------------------- Binary Quantities ---------------------##
    # time when binary is bound
    @property
    def binary_time(self):
        return self._binary_time
    
    @binary_time.setter
    def binary_time(self, v):
        self._binary_time = v

    
    # semimajor axis of binary orbit
    @property
    def semimajor_axis(self):
        return self._semimajor_axis
    
    @semimajor_axis.setter
    def semimajor_axis(self, v):
        self._semimajor_axis = v
    

    # eccentricity of binary orbit
    @property
    def eccentricity(self):
        return self._eccentricity
    
    @eccentricity.setter
    def eccentricity(self, v):
        self._eccentricity = v
    

    ##--------------------- Merger Quantities ---------------------##

    # properties of the merger remnant
    @property
    def merger_remnant(self):
        return self._merger_remnant
    
    @merger_remnant.setter
    def merger_remnant(self, v):
        self._merger_remnant = v
    

    ##--------------------- Galaxy Quantities ---------------------##

    # time of snapshot
    @property
    def analysed_snapshots(self):
        return self._analysed_snapshots
    
    @analysed_snapshots.setter
    def analysed_snapshots(self, v):
        self._analysed_snapshots = v

    
    # radial edges used in binning
    @property
    def radial_edges(self):
        return self._radial_edges
    
    @radial_edges.setter
    def radial_edges(self, v):
        self._radial_edges = v
    

    # time of snapshot
    @property
    def time_of_snapshot(self):
        return self._time_of_snapshot
    
    @time_of_snapshot.setter
    def time_of_snapshot(self, v):
        self._time_of_snapshot = v
    

    # time of snapshot
    @property
    def semimajor_axis_of_snapshot(self):
        return self._semimajor_axis_of_snapshot
    
    @semimajor_axis_of_snapshot.setter
    def semimajor_axis_of_snapshot(self, v):
        self._semimajor_axis_of_snapshot = v


    # influence radius of binary as a function of time
    @property
    def influence_radius(self):
        return self._influence_radius
    
    @influence_radius.setter
    def influence_radius(self, v):
        self._influence_radius = v
    

    # hardening radius of binary as a function of time
    @property
    def hardening_radius(self):
        return self._hardening_radius
    
    @hardening_radius.setter
    def hardening_radius(self, v):
        self._hardening_radius = v
    

    # projected mass density as a function of time
    @property
    def projected_mass_density(self):
        return self._projected_mass_density
    
    @projected_mass_density.setter
    def projected_mass_density(self, v):
        self._projected_mass_density = v
    

    # projected velocity dispersion squared (as a function of radius) as a function of time
    @property
    def projected_vel_dispersion_2(self):
        return self._projected_vel_dispersion_2
    
    @projected_vel_dispersion_2.setter
    def projected_vel_dispersion_2(self, v):
        self._projected_vel_dispersion_2 = v
    

    # effective (half-light) radius as a function of time
    @property
    def effective_radius(self):
        return self._effective_radius
    
    @effective_radius.setter
    def effective_radius(self, v):
        self._effective_radius = v
    

    # velocity dispersion squared within 1 effective radius as a function of time
    @property
    def vel_dispersion_1Re_2(self):
        return self._vel_dispersion_1Re_2
    
    @vel_dispersion_1Re_2.setter
    def vel_dispersion_1Re_2(self, v):
        self._vel_dispersion_1Re_2 = v
    

    # half mass radius as a function of time, max radius = galaxy_radius
    @property
    def half_mass_radius(self):
        return self._half_mass_radius
    
    @half_mass_radius.setter
    def half_mass_radius(self, v):
        self._half_mass_radius = v
    

    # virial mass as a function of time
    @property
    def virial_mass(self):
        return self._virial_mass
    
    @virial_mass.setter
    def virial_mass(self, v):
        self._virial_mass = v
    

    # virial radius as a function of time
    @property
    def virial_radius(self):
        return self._virial_radius
    
    @virial_radius.setter
    def virial_radius(self, v):
        self._virial_radius = v
    

    # fraction of DM within 1 effective radius as a function of time
    @property
    def inner_DM_fraction(self):
        return self._inner_DM_fraction
    
    @inner_DM_fraction.setter
    def inner_DM_fraction(self, v):
        self._inner_DM_fraction = v
    

    # velocity anisotropy (beta(r)) as a function of time
    @property
    def velocity_anisotropy(self):
        return self._velocity_anisotropy
    
    @velocity_anisotropy.setter
    def velocity_anisotropy(self, v):
        self._velocity_anisotropy = v
    

    # stellar, DM, and BH mass within galaxy_radius of centre, as a function of time
    @property
    def masses_in_galaxy_radius(self):
        return self._masses_in_galaxy_radius
    
    @masses_in_galaxy_radius.setter
    def masses_in_galaxy_radius(self, v):
        self._masses_in_galaxy_radius = v
    

    # some general functions
    def get_idx_in_vec(self, t, tarr):
        """
        Get the index of a value within an array

        Parameters
        ----------
        t : int, float
            value to search for
        tarr : array-like
            array to search within

        Returns
        -------
        int
            index of t in tarr

        Raises
        ------
        ValueError
            if value to search for is less than the first element of the array
        ValueError
            if value to search for is more than the last element of the array
        """
        try:
            assert not np.isnan(t)
        except AssertionError:
            _logger.logger.exception("t must not be nan", exc_info=True)
            raise
        try:
            idx = np.nanargmin(np.abs(tarr-t))
            if idx == len(tarr)-1:
                s = "large"
                raise AssertionError
            elif idx == 0:
                s = "smalle"
                raise AssertionError
            else:
                return idx
        except AssertionError:
            _logger.logger.exception(f"Value is {s}r than the {s}st array value!", exc_info=True)
            raise
        except ValueError:
            _logger.logger.exception(f"Array tarr has value {np.unique(tarr)}")
            raise
