import argparse

__all__ = ['argparse_for_initialise']


def argparse_for_initialise(description='', update_help=None):
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
    parser.add_argument(type=str, help='path to parameter file', dest='paramFile')
    if update_help is not None:
        parser.add_argument('-u', '--update', dest='parameter_update', help=update_help, action='store_true', default=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose printing in script')
    return parser
