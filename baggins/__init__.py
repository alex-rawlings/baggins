__all__ = [
    "analysis",
    "cosmology",
    "general",
    "initialise",
    "mathematics",
    "literature",
    "plotting",
    "utils",
    "visualisation",
    "setup_logger",
]

import baggins.analysis
import baggins.cosmology
import baggins.general
import baggins.initialise
import baggins.mathematics
import baggins.literature
import baggins.plotting
import baggins.utils
import baggins._backend
import baggins.env_config

# set up some global environment variables
HOME = baggins.env_config.home_dir
FIGDIR = baggins.env_config.figure_dir
DATADIR = baggins.env_config.data_dir
VERBOSITY = baggins.env_config.logger_level
