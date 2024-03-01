import os
import sys
from matplotlib import rc_file, rcdefaults
import subprocess
import yaml
from cmdstanpy import set_cmdstan_path
from ._backend import setup_logger
from ._backend.States import TmpDirRegister


__all__ = [
    "this_dir",
    "home_dir",
    "figure_dir",
    "data_dir",
    "date_format",
    "fig_ext",
    "TMPDIRs",
    "username",
    "git_hash",
    "_cmlogger",
]

# set up some aliases
this_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
env_params_file = os.path.join(this_dir, "env_params.yml")
with open(env_params_file, "r") as f:
    user_params, internal_params = yaml.safe_load_all(f)
figure_dir = os.path.join(home_dir, user_params["figure_dir"])
data_dir = user_params["data_dir"]
date_format = user_params["date_format"]
fig_ext = user_params["figure_ext"].lstrip(".")

# set up the temporary directory register
TMPDIRs = TmpDirRegister(os.path.join(data_dir, user_params["tmp_dir"]))

# set the stan path
set_cmdstan_path(user_params["cmdstan"])


username = home_dir.rstrip("/").split("/")[-1]

# create the logger
_cmlogger = setup_logger(
    name="cm_funcs",
    console_level=user_params["logging"]["console_level"],
    logfile=os.path.join(this_dir, user_params["logging"]["file"]),
    file_level=user_params["logging"]["file_level"],
)
# set the valid logger levels
logger_level = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ensure valid figure format
try:
    assert fig_ext in ("pdf", "png")
except AssertionError:
    _cmlogger.exception(
        f"Invalid figure format {fig_ext} given in env_params.yml! Reverting to default format: png"
    )
    fig_ext = "png"

# make the figure directory
os.makedirs(figure_dir, exist_ok=True)

# set the matplotlib settings
rcdefaults()
rc_file(os.path.join(this_dir, "plotting/matplotlibrc"))

# get the git hash
# only set git hash if in the collisionless-merger-sample repo
if "collisionless-merger-sample" in os.getcwd():
    # the standard git describe command, save git hash to yml file for use
    # of cm_functions outside the collisionless-merger-sample repo
    git_hash = (
        subprocess.run(
            ["git", "describe", "--always", "--long", "--all"],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .rstrip("\n")
    )
    internal_params["git_hash"] = git_hash
    # make updates to env_params.yml file
    with open(env_params_file, "w") as f:
        yaml.safe_dump_all(
            [user_params, internal_params], f, explicit_end=True, explicit_start=True
        )
else:
    _cmlogger.warning("Operating outside the git repo. Git hash read from file.")
    git_hash = internal_params["git_hash"]


# make sure we are using a high enough version of python
assert sys.version_info >= (3, 8, 6), "Required python version >= 3.8.6"

# purely for fun # TODO print this to screen if verbose?
"""
888888b.         d8888  .d8888b.   .d8888b.  d8b           .d8888b.  
888  "88b       d88888 d88P  Y88b d88P  Y88b Y8P          d88P  Y88b 
888  .88P      d88P888 888    888 888    888              Y88b.      
8888888K.     d88P 888 888        888        888 88888b.   "Y888b.   
888  "Y88b   d88P  888 888  88888 888  88888 888 888 "88b     "Y88b. 
888    888  d88P   888 888    888 888    888 888 888  888       "888 
888   d88P d8888888888 Y88b  d88P Y88b  d88P 888 888  888 Y88b  d88P 
8888888P" d88P     888  "Y8888P88  "Y8888P88 888 888  888  "Y8888P"  
                                                                     
                                                                     
                                                                     
"""
