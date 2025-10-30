import os
import sys
import warnings
from matplotlib import rc_file, rcdefaults
import subprocess
import yaml
from cmdstanpy import set_cmdstan_path
from baggins._backend.Logging import setup_logger
from baggins._backend.States import TmpDirRegister
from baggins._backend.CustomWarnings import ExperimentalAPIWarning


__all__ = [
    "baggins_dir",
    "home_dir",
    "figure_dir",
    "data_dir",
    "date_format",
    "fig_ext",
    "synthesizer_data",
    "TMPDIRs",
    "username",
    "git_hash",
    "_cmlogger",
]

warnings.simplefilter("always", ExperimentalAPIWarning)

# set up some aliases
baggins_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
env_params_file = os.path.join(baggins_dir, "env_params.yml")
with open(env_params_file, "r") as f:
    user_params, internal_params = yaml.safe_load_all(f)
figure_dir = os.path.join(home_dir, user_params["figure_dir"])
data_dir = user_params["data_dir"]
date_format = user_params["date_format"]
fig_ext = user_params["figure_ext"].lstrip(".")
synthesizer_data = user_params["synthesizer_data"]

# make sure we have valid paths
for p in (figure_dir, data_dir, synthesizer_data):
    assert os.path.isdir(p)

# set up the temporary directory register
TMPDIRs = TmpDirRegister(os.path.join(data_dir, user_params["tmp_dir"]))

# set the stan path
set_cmdstan_path(user_params["cmdstan"])

username = os.path.basename(home_dir)

# create the logger
os.makedirs(
    os.path.dirname(os.path.join(baggins_dir, user_params["logging"]["file"])),
    exist_ok=True,
)
_cmlogger = setup_logger(
    name="baggins",
    console_level=user_params["logging"]["console_level"],
    logfile=os.path.join(baggins_dir, user_params["logging"]["file"]),
    file_level=user_params["logging"]["file_level"],
)
# set the valid logger levels
logger_level = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# determine the usage mode: dev or user
usage_mode = user_params["mode"]
if usage_mode != "dev" and usage_mode != "user":
    _cmlogger.warning(
        f"Invalid usage mode '{usage_mode}' given, reverting to mode 'user'."
    )
    usage_mode = "user"

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
rc_file(os.path.join(baggins_dir, "plotting/matplotlibrc"))

# get the git hash
# only set git hash if in the collisionless-merger-sample repo
if "collisionless-merger-sample" in os.getcwd():
    # the standard git describe command, save git hash to yml file for use
    # of baggins outside the collisionless-merger-sample repo
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
    if usage_mode == "dev":
        _cmlogger.warning("Operating outside the git repo. Git hash read from file.")
    git_hash = internal_params["git_hash"]


# make sure we are using a high enough version of python
assert sys.version_info >= (3, 8, 6), "Required python version >= 3.8.6"
