import os
import sys
from matplotlib import rc_file, rcdefaults
import subprocess
import json
from datetime import datetime
from .utils.log_info import CustomLogger


__all__ = ["this_dir", "home_dir", "figure_dir", "data_dir", "date_format", "username", "git_hash", "cmf_logger"]


this_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
env_params_file = os.path.join(this_dir, "env_params.json")
with open(env_params_file, "r") as f:
    env_params = json.load(f)
figure_dir = os.path.join(home_dir, env_params["user_settings"]["figure_dir"])
data_dir = env_params["user_settings"]["data_dir"]
date_format = env_params["user_settings"]["date_format"]

username = home_dir.rstrip("/").split("/")[-1]

# create the logger
cmf_logger = CustomLogger("cm_funcs", env_params["user_settings"]["logging"]["console_level"])
if env_params["user_settings"]["logging"]["file_level"] not in ["", " "]:
    lf = f"{env_params['user_settings']['logging']['file'].rstrip('.log')}_{datetime.now():%Y-%m-%d}.log"
    cmf_logger.add_file_handler(os.path.join(this_dir, lf), env_params["user_settings"]["logging"]["file_level"])

#make the figure directory
os.makedirs(figure_dir, exist_ok=True)

#set the matplotlib settings
rcdefaults()
rc_file(os.path.join(this_dir, "plotting/matplotlibrc"))

# get the git hash
# only set git hash if in the collisionless-merger-sample repo
if "collisionless-merger-sample" in os.getcwd():
    # the standard git describe command, save git hash to json file for use 
    # of cm_functions outside the collisionless-merger-sample repo
    git_hash = subprocess.run(["git", "describe", "--always", "--long", "--all"], check=True, capture_output=True).stdout.decode().rstrip("\n")
    env_params["internal_settings"]["git_hash"] = git_hash
    # make updates to env_params.json file
    with open(env_params_file, "w") as f:
        json.dump(env_params, f, indent=4)
else:
    cmf_logger.logger.warning("Operating outside the git repo. Git hash read from file.")
    git_hash = env_params["internal_settings"]["git_hash"]


#make sure we are using a high enough version of python
assert sys.version_info >= (3,8,6), "Required python version >= 3.8.6"