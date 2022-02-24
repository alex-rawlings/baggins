import os
import sys
import matplotlib

__all__ = ["this_dir", "home_dir", "figure_dir", "username"]

this_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
figure_dir = os.path.join(home_dir, "figures")

username = home_dir.rstrip("/").split("/")[-1]

#make the figure directory
os.makedirs(figure_dir, exist_ok=True)

#set the matplotlib settings
matplotlib.rcdefaults()
matplotlib.rc_file(os.path.join(this_dir, "plotting/matplotlibrc"))


#make sure we are using a high enough version of python
assert sys.version_info >= (3,8,6), "Required python version >= 3.8.6"