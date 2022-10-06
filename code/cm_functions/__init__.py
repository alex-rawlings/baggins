from . import env_config

from . import analysis
from . import cosmology
from . import general
from . import initialise
from . import mathematics
from . import literature
from . import plotting
from . import utils
from . import visualisation

from ._backend import CustomLogger

#set up some global environment variables
HOME = env_config.home_dir
FIGDIR = env_config.figure_dir
DATADIR = env_config.data_dir
VERBOSITY = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]