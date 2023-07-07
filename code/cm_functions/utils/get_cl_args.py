import argparse
import numpy as np
from .. import VERBOSITY

__all__ = ["argparse_for_initialise", "argparse_for_stan", "cl_str_2_space"]


def argparse_for_initialise(description="", update_help=None):
    """
    Get the command line arguments necessary to run the initialisation scripts,
    as they all have a similar format with similar options.

    Parameters
    ----------
    description : str, optional
        main description of program invoked with help flag, by default ""
    update_help : str, optional
        description of which parameters can be updated. A value of None 
        excludes the argument, by default None

    Returns
    -------
    argparse.ArgumentParser
        argument parser for other arguments specific to the script to be parsed
        to
    """
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(type=str, help="path to parameter file", dest="paramFile")
    if update_help is not None:
        parser.add_argument("-u", "--update", dest="parameter_update", help=update_help, action="store_true", default=False)
    return parser


def argparse_for_stan(description=""):
    """
    Get the command line arguments necessary to run Stan models, as they are 
    all similar.

    Parameters
    ----------
    description : str, optional
        main description of program invoked with help flag, by default ""

    Returns
    -------
    argparse.ArgumentParser
        argument parser for other arguments specific to the script to be parsed
        to
    """
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
    parser.add_argument(type=str, help="directory to HMQuantity HDF5 files or csv files", dest="dir")
    parser.add_argument(type=str, help="new sample or load previous", choices=["new", "loaded"], dest="type")
    parser.add_argument("-p", "--prior", help="plot for prior", action="store_true", dest="prior")
    parser.add_argument("-s", "--sample", help="sample set", type=str, dest="sample", choices=["mcs", "perturb"], default="mcs")
    parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
    parser.add_argument("-N", "--NumSamples", type=int, help="number OOS values", dest="NOOS", default=1000)
    parser.add_argument("-v", "--verbosity", type=str, choices=VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
    return parser


def cl_str_2_space(s, space_type="lin"):
    """
    Convert a string representation of a list to a numpy.linspace, numpy.
    logspace, or numpy.geomspace instance. The string must be of the form "[a,b,
    c]", where a is the lower value, b is the upper value, and c is the number 
    of elements in the space. For compatability with command line arguments 
    (which is where this function is primarily designed to be used) there 
    should be no spaces between the elements.

    Parameters
    ----------
    s : str
        string to convert
    space_type : str, optional
        lin, log, or geom, spacing type, by default "lin"

    Returns
    -------
    : np.ndarray
        array with specified spacing

    Raises
    ------
    ValueError
        invalid space_type input
    """
    char_arr = list(s)
    #make sure the input "looks" like a list
    assert(char_arr[0] == "[" and char_arr[-1]=="]")
    s = s.strip("[]")
    num_arr = s.split(",")
    assert(len(num_arr)==3)
    if space_type == "lin":
        return np.linspace(float(num_arr[0]), float(num_arr[1]), int(num_arr[2]))
    elif space_type == "log":
        return np.logspace(float(num_arr[0]), float(num_arr[1]), int(num_arr[2]))
    elif space_type == "geom":
        return np.geomspace(float(num_arr[0]), float(num_arr[1]), int(num_arr[2]))
    else:
        raise ValueError("The spacing type must be either lin, log, or geom!")