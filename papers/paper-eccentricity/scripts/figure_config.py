import os
from matplotlib import rc_file, rcdefaults
from Plotter import Plotter

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(this_dir, 'data/'))
figure_dir = os.path.abspath(os.path.join(this_dir, '../figures/'))

# get the complete path for saving a figure
def fig_path(fname):
    return os.path.join(figure_dir, fname)

# get the complete path to a saved data file
def data_path(dname):
    return os.path.join(data_dir, dname)

# make sure were using the same settings
rcdefaults()
rc_file(os.path.join(this_dir,'matplotlibrc_publish'))

# set up a custom plotting object
plotter = Plotter()