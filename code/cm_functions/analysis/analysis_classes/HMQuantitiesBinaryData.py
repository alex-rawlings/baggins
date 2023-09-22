import warnings
import h5py
from . import HMQuantitiesSingleData
from ...general import get_idx_in_array
from ...env_config import _cmlogger


_logger = _cmlogger.copy(__file__)

__all__ = ["HMQuantitiesBinaryData"]


class HMQuantitiesBinaryData(HMQuantitiesSingleData):
    def __init__(self) -> None:
        """
        Base class representing the quantities of a hierarchical model. Allows 
        for easy saving and loading of the class properties to and from a HDF5 
        file.
        The class includes information about the BH binary.
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
    
    # angular momentum of binary
    @property
    def binary_angular_momentum(self):
        return self._binary_angular_momentum
    
    @binary_angular_momentum.setter
    def binary_angular_momentum(self, v):
        self._binary_angular_momentum = v
    
    # energy of binary
    @property
    def binary_energy(self):
        return self._binary_energy
    
    @binary_energy.setter
    def binary_energy(self, v):
        self._binary_energy = v
    
    # period of binary
    @property
    def binary_period(self):
        return self._binary_period
    
    @binary_period.setter
    def binary_period(self, v):
        self._binary_period = v

    # separation of BHs in binary
    @property
    def binary_separation(self):
        return self._binary_separation

    @binary_separation.setter
    def binary_separation(self, v):
        self._binary_separation = v
    
    # BH masses
    @property
    def binary_masses(self):
        return self._binary_masses

    @binary_masses.setter
    def binary_masses(self, v):
        self._binary_masses = v

    # pericentre deflection angles before binary is bound
    @property
    def prebound_deflection_angles(self):
        return self._prebound_deflection_angles

    @prebound_deflection_angles.setter
    def prebound_deflection_angles(self, v):
        self._prebound_deflection_angles = v


    ##--------------------- Galaxy Binary Quantities ---------------------##

    # semimajor axis of binary in snapshot
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

    # G*rho/sigma as a function of time
    @property
    def G_rho_per_sigma(self):
        return self._G_rho_per_sigma
    
    @G_rho_per_sigma.setter
    def G_rho_per_sigma(self, v):
        self._G_rho_per_sigma = v


    ##--------------------- Some General Functions ---------------------##
    def get_idx_in_vec(self, t, tarr):
        warnings.warn("This function should be called using idx_finder()", DeprecationWarning)
        return self.idx_finder(t, tarr)


    def idx_finder(self, val, vec):
        """
        Wrapper around general.get_idx_in_array(), better suited for data from
        this class

        Parameters
        ----------
        val : float
            value to search for
        vec : array-like
            array to search in

        Returns
        -------
        status : bool
            was search successful
        idx : int
            index of val in vec
        """
        try:
            idx = get_idx_in_array(val, vec)
            status = True
        except ValueError:
            _logger.logger.warning(f"No data prior to merger! The requested semimajor axis value is {val}, semimajor_axis attribute is: {vec}. This run will not form part of the analysis.")
            status = False
            idx = -9999
        except AssertionError:
            _logger.logger.warning(f"Trying to search for value {val}, but an AssertionError was thrown. The array bounds are {min(vec)} - {max(vec)}. This run will not form part of the analysis.")
            status = False
            idx = -9999
        return status, idx