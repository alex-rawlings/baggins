import os
import sys
import matplotlib

__all__ = ["this_dir", "home_dir", "figure_dir"]

this_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
figure_dir = os.path.join(home_dir, "figures")

#make the figure directory
os.makedirs(figure_dir, exist_ok=True)

#set the matplotlib settings
matplotlib.rcdefaults()
matplotlib.rc_file(os.path.join(this_dir, "plotting/matplotlibrc"))
assert sys.version_info >= (3,8,6), "Required python version >= 3.8.6"