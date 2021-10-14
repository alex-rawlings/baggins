import argparse
import numpy as np

__all__ = ["argparse_for_initialise", "cl_str_2_space"]


def argparse_for_initialise(description="", update_help=None):
    """
    Get the command line arguments necessary to run the initialisation scripts,
    as they all have a similar format with similar options.

    Parameters
    ----------
    description: main description of program invoked with help flag
    update_help: description of which parameters can be updated. A value
                 of None excludes the argument

    Returns
    -------
    parser object that can have other options added to it
    """
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(type=str, help="path to parameter file", dest="paramFile")
    if update_help is not None:
        parser.add_argument("-u", "--update", dest="parameter_update", help=update_help, action="store_true", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose printing in script")
    return parser


def cl_str_2_space(s, space_type="lin"):
    """
    Convert a string representation of a list to a numpy.linspace or numpy.
    logspace instance. The string must be of the form "[a,b,c]", where a is the 
    lower value, b is the upper value, and c is the number of elements in the 
    space. For compatability with command line arguments (which is where this
    function is primarily designed to be used) there should be no spaces between
    the elements.

    Parameters
    ----------
    s: string to convert
    space_type: lin or log, spacing type

    Returns
    -------
    numpy.linspace or numpy.logspace with the specified parameters
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
    else:
        raise ValueError("The spacing type must be either lin or log !")