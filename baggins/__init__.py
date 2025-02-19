__all__ = [
    "analysis",
    "cosmology",
    "general",
    "initialise",
    "mathematics",
    "literature",
    "plotting",
    "utils",
    "setup_logger",
]

import baggins.analysis as analysis
import baggins.cosmology as cosmology
import baggins.general as general
import baggins.initialise as initialise
import baggins.mathematics as mathematics
import baggins.literature as literature
import baggins.plotting as plotting
import baggins.utils as utils
import baggins._backend
import baggins.env_config

from baggins._backend import setup_logger

# set up some global environment variables
HOME = baggins.env_config.home_dir
FIGDIR = baggins.env_config.figure_dir
DATADIR = baggins.env_config.data_dir
VERBOSITY = baggins.env_config.logger_level
