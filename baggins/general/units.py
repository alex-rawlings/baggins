import numpy as np
import ketjugw.units
import pygad


__all__ = ["kpc", "Myr", "Gyr", "NBodyUnits"]

"""
ketjugw conversions
"""
kpc = 1e3 * ketjugw.units.pc
Myr = 1e6 * ketjugw.units.yr
Gyr = 1e9 * ketjugw.units.yr


class NBodyUnits:
    def __init__(self, unit_length_in_kpc, unit_mass_in_Msol) -> None:
        """
        N-body unit conversions, courtesy Matias

        Parameters
        ----------
        unit_length_in_kpc : float
            unit length of system in kpc
        unit_mass_in_Msol : float
            unit mass of system in solar masses
        """
        self._unit_length_in_kpc = pygad.UnitScalar(unit_length_in_kpc, "kpc")
        self._unit_mass_in_Msol = pygad.UnitScalar(unit_mass_in_Msol, "Msol")
        self._unit_velocity = np.sqrt(
            pygad.physics.G * self.unit_mass_in_Msol / self.unit_length_in_kpc
        )
        self._unit_time = self.unit_length_in_kpc / self.unit_velocity
        self._unit_J = self.unit_length_in_kpc * self.unit_velocity

    @property
    def unit_length_in_kpc(self):
        return self._unit_length_in_kpc

    @property
    def unit_mass_in_Msol(self):
        return self._unit_mass_in_Msol

    @property
    def unit_velocity(self):
        return self._unit_velocity

    @property
    def unit_time(self):
        return self._unit_time

    @property
    def unit_J(self):
        return self._unit_J
