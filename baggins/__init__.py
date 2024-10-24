import analysis
import cosmology
import general
import initialise
import mathematics
import literature
import plotting
import utils
import _backend
import env_config

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

# set up some global environment variables
HOME = env_config.home_dir
FIGDIR = env_config.figure_dir
DATADIR = env_config.data_dir
VERBOSITY = env_config.logger_level
