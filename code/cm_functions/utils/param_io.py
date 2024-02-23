import numpy as np
import json
import yaml
from pygad import UnitArr

from ..env_config import _cmlogger

_logger = _cmlogger.getChild(__name__)


__all__ = [
    "read_parameters",
    "write_calculated_parameters",
    "overwrite_parameter_file",
    "to_json",
]


class ScientificDumper(yaml.SafeDumper):
    def represent_float(self, data):
        """
        Overload the default YAML SafeDumper float representation method to
        control when we start to use scientific notation

        Parameters
        ----------
        data : float
            data to represent

        Returns
        -------
        pyyaml.ScalarNode
            method for representing floats
        """
        if data != data or (data == 0.0 and data == 1.0):
            value = ".nan"
        elif data == self.inf_value:
            value = ".inf"
        elif data == -self.inf_value:
            value = "-.inf"
        else:
            if data < 1e4:
                value = repr(data).lower()
            else:
                value = f"{data:.8e}".lower()
            if "." not in value and "e" in value:
                value = value.replace("e", ".0e", 1)
        return self.represent_scalar("tag:yaml.org,2002:float", value)


def read_parameters(filepath):
    """
    Read a .yml configuration file. Numpy methods can be specified and saved to
    the loaded dictionary.

    Parameters
    ----------
    filepath : str, path-like
        path to parameter file to read

    Returns
    -------
    params_and_calc : dict
        dictionary of user parameters and calculated parameters (the latter stored under the key top-level key 'calculated')
    """

    def _unpack_helper(d, lev):
        """
        Helper to unpack a numpy method, e.g. a linspace

        Parameters
        ----------
        d : dict
            dictionary to unpack
        lev : int
            level of unpacking (to keep track of reserved top-level keys)

        Returns
        -------
        d : dict
            dictionary with the results of a numpy method saved to it
        """
        for k, v in d.copy().items():
            try:
                if lev == 0:
                    assert k != "calculated"
            except AssertionError:
                _logger.exception(
                    "Main parameter block cannot contain the top-level reserved key 'calculated'!",
                    exc_info=True,
                )
                raise
            if k == "numpy_method":
                try:
                    args = d["args"]
                    if not isinstance(args, list):
                        args = [args]
                except KeyError:
                    _logger.exception(
                        f"Error reading parameter file {filepath}! Blocks with key 'numpy_method' must have a corresponding 'args' key!",
                        exc_info=True,
                    )
                    raise
                try:
                    assert "value" not in d
                except AssertionError:
                    _logger.exception(
                        f"Error reading parameter file {filepath}! Blocks with key 'numpy_method' must not have a corresponding 'value' key!",
                        exc_info=True,
                    )
                    raise
                method = getattr(np, v)
                d["value"] = method(*args)
            elif isinstance(v, str) and v[-1] == "/":
                d[k] = v.rstrip("/")
            elif isinstance(v, dict):
                lev += 1
                _unpack_helper(v, lev)
                lev -= 1
        return d

    with open(filepath, "r") as f:
        params_list = list(yaml.safe_load_all(f))
    params_and_calc = params_list[0].copy()
    params_and_calc["calculated"] = {}
    for i, params in enumerate(params_list):
        params = _unpack_helper(params, l=0)
        if i == 0:
            continue
        for k in params.keys():
            params_and_calc["calculated"][k] = params[k]
    return params_and_calc


def write_calculated_parameters(data, filepath):
    """
    Write static, calculated variables (e.g. particle count) to a new .yml
    block located within the .yml file (by using the --- separators)
    corresponding to the system that was created. Parameters in the "main"
    block are never overwritten. Previously-calculated values may be
    overwritten. New values will be added to the second block.

    Parameters
    ----------
    data : dict
        parameters (name, value) to add to the parameter file
    filepath : str, path-like
        path to parameter file where values will be saved
    """

    def _type_converter(d, d2):
        new_d = d2.copy()
        for k, v in d.items():
            if isinstance(v, (np.float64, np.float32)):
                new_d[k] = float(v)
            elif isinstance(v, np.ndarray):
                new_d[k] = v.tolist()
            elif isinstance(v, UnitArr):
                new_d[k] = {"unit": str(v.units).strip("[]"), "value": float(v)}
            elif isinstance(v, dict):
                try:
                    new_d[k] = _type_converter(v, new_d[k])
                except KeyError:
                    new_d[k] = _type_converter(v, {})
            else:
                new_d[k] = d[k]
        return new_d

    with open(filepath, "r+") as f:
        _data_list = list(yaml.safe_load_all(f))
        # skip the first block, which consists of the user-defined parameters
        if len(_data_list) == 1:
            _data_list.append(data)
        # update values: will update all blocks after the first
        _data_list[1] = _type_converter(data, _data_list[1])
        f.seek(0)
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            f.write(line)
            # we have reached the end of the parameter section
            if "..." in line:
                break
        f.write("\n")
        s = yaml.dump_all(
            _data_list[1:],
            explicit_end=True,
            explicit_start=True,
            Dumper=ScientificDumper,
        )
        f.write(s)
        f.write("\n")
        f.truncate()


def overwrite_parameter_file(f, contents):
    """
    Overwrite a .yml parameter file, with only the first document being written
    if there are multiple .yml documents in the file. Explicit endings ('...')
    are always printed.

    Parameters
    ----------
    f : TextIOWrapper
        file pointer to the file to overwrite
    contents : str
        contents to write to file
    """
    # write the new contents
    f.seek(0)
    f.write(contents)
    f.truncate()
    # remove calculated quantities
    f.seek(0)
    lines = f.readlines()
    f.seek(0)
    f.truncate()
    for line in lines:
        f.write(line)
        if line == "...\n":
            break
    else:
        f.write("...\n")


def to_json(obj, fname):
    """
    Convert a .py parameter file to a .json representation

    Parameters
    ----------
    obj : dict
        object to serialise
    fname : str, path-like
        file name to save to
    """
    d = {}
    for k, v in obj.items():
        if k[:2] != "__":
            if isinstance(v, np.ndarray):
                v = v.tolist()
            d[k] = v
    with open(fname, "w") as f:
        json.dump(d, f, indent=4)
