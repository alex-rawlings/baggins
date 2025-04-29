import os.path
import numpy as np
import scipy.optimize
import scipy.special
from time import time
from baggins.env_config import _cmlogger
from baggins.cosmology import luminosity_distance, angular_diameter_distance

__all__ = [
    "BasicQuantityConverter",
    "sersic_b_param",
    "xval_of_quantity",
    "set_seed_time",
    "get_idx_in_array",
    "get_unique_path_part",
    "represent_numeric_in_scientific",
    "get_snapshot_number",
    "print_dict_summary",
]

_logger = _cmlogger.getChild(__name__)


class BasicQuantityConverter:
    def __init__(
        self,
        z=None,
        dist=None,
        dist_mod=None,
        abs_mag=None,
        app_mag=None,
        mass_light_ratio=4.0,
    ):
        """
        Class to do conversions of basic astronomical quantities

        Parameters
        ----------
        z : float, optional
            redshift (will be used to determine a luminosity distance), by default None
        dist : float, optional
            distance, by default None
        dist_mod : float, optional
            distance modulus, by default None
        abs_mag : float, optional
            absolute magnitude, by default None
        app_mag : float, optional
            apparent magnitude, by default None
        mass_light_ratio : float, optional
            mass-light ratio, by default 4.
        """
        if z is not None:
            _logger.info("Redshift given, 'dist' and 'dist_mod' will be ignored")
            if isinstance(z, (float, int)):
                z = np.array([z])
            self.z = z
            self.dist = np.full_like(z, np.nan)
            self.dist_ang = np.full_like(z, np.nan)
            for i, zz in enumerate(z):
                self.dist[i] = luminosity_distance(zz) * 1e3
                self.dist_ang[i] = angular_diameter_distance(zz) * 1e3
            self._convert_pc_to_modulus()
        elif dist is not None:
            _logger.info("Distance given, 'z' and 'dist_mod' will be ignored")
            self.z = np.zeros_like(dist)
            self.dist = dist
            self.dist_ang = dist
            self._convert_pc_to_modulus()
        else:
            _logger.info("Distance modulus given, 'z' and 'dist' will be ignored")
            self.z = np.zeros_like(dist)
            self.dist_mod = dist_mod
            self._convert_modulus_to_pc()
            self.dist_ang = self.dist
        if abs_mag is not None:
            _logger.info("Absolute magnitude given, 'app_mag' will be ignored")
            self.abs_mag = abs_mag
            self.app_mag = self.convert_abs_mag_to_app_mag(self.abs_mag)
        else:
            _logger.info("Apparent magnitude given, 'abs_mag' will be ignored")
            self.app_mag = app_mag
            self.abs_mag = self.convert_app_mag_to_abs_mag(self.app_mag)
        self.mass_light_ratio = mass_light_ratio
        self.lum = self.convert_mag_to_lum(self.abs_mag)
        self.mass = self.convert_lum_to_mass(self.lum)

    @property
    def abs_mag_solar(self):
        return 4.83

    def convert_app_mag_to_abs_mag(self, m):
        return m - self.dist_mod

    def convert_abs_mag_to_app_mag(self, M):
        return M + self.dist_mod

    def _convert_modulus_to_pc(self):
        self.dist = 10 ** (1 + self.dist_mod / 5)

    def _convert_pc_to_modulus(self):
        self.dist_mod = 5 * (np.log10(self.dist) - 1)

    def convert_mag_to_lum(self, M):
        return 10 ** ((M - self.abs_mag_solar) / -2.5)

    def convert_lum_to_mass(self, L, mlr=None):
        if mlr is None:
            mlr = self.mass_light_ratio
        return L * mlr

    def convert_ang_size_to_pc(self, t):
        return np.tan(t * np.pi / (3600 * 180)) * self.dist_ang


def sersic_b_param(n):
    """
    Determine the b parameter in the Sersic function
    search interval given by n=[0.5,20] -> 2n-0.33+0.009876/n

    Parameters
    ----------
    n : float
        sersic index

    Returns
    -------
    : float
        sersic b parameter
    """
    return scipy.optimize.toms748(
        lambda t: 2 * scipy.special.gammainc(2 * n, t) - 1, 0.6, 19.9
    )


def xval_of_quantity(val, xvec, yvec, initial_guess=None, root_kwargs={}):
    """
    Find the value in a set of independent observations corresponding to a
    dependent observation. For example, the time corresponding to a particular
    radius value. Linear interpolation is done to create a function
    y = f(x), on which root finding is performed.

    Parameters
    ----------
    val : float
        y-value to determine the corresponding x-value for
    xvec : np.ndarray
        independent observations
    yvec : np.ndarray
        dependent observations
    initial_guess : list, optional
        [a,b], where a and b specify the bounds within which val should occur.
        Must be "either side" of val. By default None, sets [a, b] = [xvec[0],
        xvec[-1]]
    root_kwargs : dict, optional
        other keyword arguments to be parsed to the root finding algorithm
        (scipy.optimize.brentq), by default {}

    Returns
    -------
    xval : float
        value of independent observations corresponding to the observed
        dependent observation value
    """
    try:
        assert np.all(np.diff(xvec) > 0)
    except AssertionError:
        _logger.exception("x values must be ordered and increasing!", exc_info=True)
        raise
    # create the linear interpolating function
    f = lambda x: np.interp(x, xvec, yvec - val)
    if initial_guess is None:
        initial_guess = [xvec[0], xvec[-1]]
    xval, rootresult = scipy.optimize.brentq(
        f, *initial_guess, full_output=True, **root_kwargs
    )
    if not rootresult.converged:
        # TODO should the method terminate instead?
        _logger.warning(
            f"The root-finding did not converge after {rootresult.iterations} iterations! The input <val> may not be in the domain specified by <xvec>."
        )
    return xval


def set_seed_time():
    """
    Create a random number generator seed that is the inverse of the current
    time.

    Returns
    -------
    : int
        seed
    """
    s = f"{int(time())}"[::-1]
    return int(s)


def get_idx_in_array(t, tarr):
    """
    Get the index of a value within an array. If multiple matches are found,
    the first is returned (following np.argmin method)

    Parameters
    ----------
    t : int, float
        value to search for
    tarr : array-like
        array to search within

    Returns
    -------
    idx : int
        index of t in tarr

    Raises
    ------
    AssertionError
        value to search for is a nan
    AssertionError
        if index is 0 or the last index of the array
    """
    try:
        assert not np.isnan(t)
    except AssertionError:
        _logger.exception("t must not be nan", exc_info=True)
        raise
    try:
        idx = np.nanargmin(np.abs(tarr - t))
        if idx == len(tarr) - 1:
            s = "large"
            raise AssertionError
        elif idx == 0:
            s = "smalle"
            raise AssertionError
        else:
            return idx
    except AssertionError:
        _logger.exception(f"Value is {s}r than the {s}st array value!", exc_info=True)
        raise
    except ValueError:
        _logger.exception(f"Array tarr has value {np.unique(tarr)}")
        raise


def get_unique_path_part(path_list):
    """
    Determine the parts of a path that are unique in a set of paths

    Parameters
    ----------
    str_list : list
        list of paths

    Returns
    -------
    unique_parts : list
        list of unique parts of the paths
    """
    try:
        assert len(path_list) > 1
    except AssertionError:
        _logger.exception("Path list must contain more than 1 path!", exc_info=True)
        raise
    common_path_len = len(os.path.commonpath(path_list))
    unique_parts = []
    for s in path_list:
        unique_parts.append(s[common_path_len:].strip("/"))
    return unique_parts


def represent_numeric_in_scientific(v, mantissa_fmt=".1f"):
    if mantissa_fmt[-1] != "f":
        _logger.error(
            "Mantissa format must be of floating point type! Using default value '.1f'"
        )
        mantissa_fmt = ".1f"
    exponent = int(np.floor(np.log10(v)))
    return f"${v/10**exponent:{mantissa_fmt}}\\times 10^{exponent}$"


def get_snapshot_number(s, prefix="snap"):
    """
    Get the number of a snapshot from the file name.

    Parameters
    ----------
    s : str, path-like
        snapshot file name
    prefix : str, optional
        prefix to remove from file name, by default "snap"

    Returns
    -------
    : str
        snapshot identifier
    """
    fname = os.path.splitext((os.path.basename(s)))[0]
    return fname.replace(f"{prefix}_", "")


def print_dict_summary(d, level=0):
    """
    Print a summary of the contents of a dictionary.
    Recursively calls itself to handle sub-dicts.

    Parameters
    ----------
    d : dict
        dictionary to print.
    level : int, optional
        spacing level, by default 0
    """
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * level + f"> {k}")
            level += 1
            print_dict_summary(v, level=level)
            level -= 1
        else:
            if isinstance(v, np.ndarray):
                extra_info = f"shape: {v.shape}"
            elif isinstance(v, list):
                extra_info = f"len: {len(v)}"
            else:
                extra_info = ""
            print(" " * level + f"> {k}: {type(v)} - {extra_info}")
