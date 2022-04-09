import re
import sys
import os
import importlib
import numpy as np


__all__ = ["read_parameters", "write_parameters"]


def read_parameters(filepath, verbose=True):
    """
    Read parameters from a python file as if it were a module. If the 
    invocation is:
        param_vals = read_parameters(file)
    then the parameter file values can be accessed as param_vals.value
    Strings have trailing '/' characters removed, and all lists are converted
    to class <np.ndarray>

    Parameters
    ----------
    filepath : str
        absolute or relative path to the parameter file
    verbose : bool, optional
        print the name of the file being read?, by default True

    Returns
    -------
    params : types.ModuleType
        imported parameter file as a module

    Raises
    ------
    ValueError
        filepath must end in .py extension
    """
    name_ext = filepath.split("/")[-1]
    sys.path.append(os.path.dirname(os.path.abspath(filepath)))
    module_name, module_ext = name_ext.split(".")
    if module_ext != "py":
        raise ValueError("Input file must have .py extension!")
    else:
        if verbose:
            print(f"Reading parameters from: {filepath}")
        params = importlib.import_module(module_name)
        #remove trailing '/' characters from strings
        for p in dir(params):
            if p[:2] == "__":
                continue
            v = getattr(params, p)
            if isinstance(v, str) and v[-1].rstrip()=="/":
                setattr(params, p, v.rstrip("/"))
            elif isinstance(v, list):
                setattr(params, p, np.array(v, dtype="float64"))
        return params


def write_parameters(values, filepath=None, verbose=True, allow_updates=()):
    """
    Write parameters that have been loaded with read_parameters() to a file.

    Parameters
    ----------
    values : types.ModuleType
        module name which contains the parameters
    filepath : str, optional
        file to save the parameters to, by default None (values.__file__ is
        used)
    verbose : bool, optional
        print the name of the file being written to?, by default True
    allow_updates : tuple, optional
        tuple of parameter names as strings that are allowed to be updated 
        (prevents accidental overwriting), by default ()
    """
    if filepath is None:
        filepath = values.__file__
    new_vars = False
    with open(filepath, "r+") as f:
        contents = f.read()
        for var in dir(values):
            if var[:2] != "__":
                value = getattr(values, var)
                #include any potential comments
                line = re.search(r"^\b{}\b.*".format(var), contents, flags=re.MULTILINE)
                if line is not None:
                    #the variable exists in the parameter file
                    if var not in allow_updates:
                        # but we don't want to update this value
                        continue
                    if verbose:
                        print(f"Updating variable: {var}")
                    if "#" in line.group(0):
                        comment = "  #" + "#".join(line.group(0).split("#")[1:])
                    else:
                        comment = ""
                    if isinstance(value, str):
                        contents = re.sub(r"^\b{}\b.*".format(var), '{} = "{}"{}'.format(var, value, comment), contents)
                    elif isinstance(value, np.ndarray):
                        value = np.array2string(value, precision=5, floatmode="maxprec", separator=",", sign="+")
                        contents = re.sub(r"^\b{}\b.*".format(var), '{} = {}{}'.format(var, value, comment), contents)
                    else:
                        contents = re.sub(r"^\b{}\b.*".format(var), '{} = {:.5e}{}'.format(var, value, comment), contents)
                else:
                    #we are adding a new value to the parameter file
                    new_vars = True
                    if verbose:
                        print(f"Adding variable: {var}")
                    if isinstance(value, str):
                        contents += '{} = "{}"\n'.format(var, value)
                    else:
                        contents += '{} = {:.5e}\n'.format(var, value)
        if new_vars:
            #add a dividing line to make it easier to see which outputs are
            #from which scripts
            contents += "#----------------------\n"
        if verbose:
            print(f"Writing parameters to: {filepath}")
        f.seek(0)
        #overwrite entire file
        f.write(contents)
        #reduce file size
        f.truncate()
