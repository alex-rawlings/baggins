import os
import sys
import matplotlib
from importlib.metadata import version
from pkg_resources import parse_version

__all__ = ["this_dir", "home_dir", "figure_dir"]

this_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
figure_dir = os.path.join(home_dir, "figures")

#make the figure directory
os.makedirs(figure_dir, exist_ok=True)

#set the matplotlib settings
matplotlib.rcdefaults()
matplotlib.rc_file(os.path.join(this_dir, "plotting/matplotlibrc"))

#check package dependencies
package_reqs = dict(
    numpy = {"lower": "1.19.5", "upper": None},
    scipy = {"lower": "1.5.4", "upper": None},
    matplotlib = {"lower": "3.3.3", "upper": None},
    pandas = {"lower": "1.3.1", "upper": None},
    h5py = {"lower": "3.3.0", "upper": None},
    seaborn = {"lower": "0.11.1", "upper": None}
)
for this_package in package_reqs.keys():
    if package_reqs[this_package]["lower"] is not None:
        assert parse_version(version(this_package)) >= parse_version(package_reqs[this_package]["lower"]), "Package {} must be >= {}".format(this_package, package_reqs[this_package]["lower"])
    if package_reqs[this_package]["upper"] is not None:
        assert parse_version(version(this_package)) <= parse_version(package_reqs[this_package]["upper"]), "Package {} must be <= {}".format(this_package, package_reqs[this_package]["upper"])
assert sys.version_info >= (3,8,6), "Required python version >= 3.8.6"