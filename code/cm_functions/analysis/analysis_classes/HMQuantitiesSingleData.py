import h5py
from . import HDF5Base
from ...env_config import _cmlogger

__all__ = []


_logger = _cmlogger.getChild(__name__)


class HMQuantitiesSingleData(HDF5Base):
    def __init__(self) -> None:
        """
        Base class representing the quantities of a hierarchical model. Allows
        for easy saving and loading of the class properties to and from a HDF5
        file.
        The class does not include information about the BH binary.
        """
        super().__init__()
        self.merger_id = None

    # #--------------------- Merger Quantities ---------------------##

    # properties of the merger remnant
    @property
    def merger_remnant(self):
        return self._merger_remnant

    @merger_remnant.setter
    def merger_remnant(self, v):
        self._merger_remnant = v

    # #--------------------- Galaxy Quantities ---------------------##

    # analysed of snapshot
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

    # particles masses of components
    @property
    def particle_masses(self):
        return self._particle_masses

    @particle_masses.setter
    def particle_masses(self, v):
        self._particle_masses = v

    # initial orbital parameters
    @property
    def initial_galaxy_orbit(self):
        return self._initial_galaxy_orbit

    @initial_galaxy_orbit.setter
    def initial_galaxy_orbit(self, v):
        self._initial_galaxy_orbit = v

    # initial orbital parameters
    @property
    def escape_velocity(self):
        return self._escape_velocity

    @escape_velocity.setter
    def escape_velocity(self, v):
        self._escape_velocity = v

    @classmethod
    def load_from_file(cls, fname, decode="utf-8"):
        """
        See docs for HDF5Base load_from_file() method
        """
        C = super().load_from_file(fname, decode=decode)
        with h5py.File(fname, "r") as f:
            C.merger_id = f["/meta"].attrs["merger_id"]
        return C

    # --------------------- Some General Functions ---------------------##
    def mass_resolution(self):
        """
        Determine the mass resolution for this simulation

        Returns
        -------
        : float
            mass resolution
        """
        if "stars" not in self.particle_masses:
            _logger.error(
                "Key 'stars' not present in 'particle_masses', trying 'dm' instead..."
            )
            try:
                field_part_mass = self.particle_masses["dm"]
            except KeyError:
                _logger.exception(
                    "Key 'dm' not present in 'particle_masses': need one of 'stars' or 'dm'!",
                    exc_info=True,
                )
                raise
        else:
            field_part_mass = self.particle_masses["stars"]
        return self.particle_masses["bh"] / field_part_mass
