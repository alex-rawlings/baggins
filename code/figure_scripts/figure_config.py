import os
import matplotlib
import sys

this_dir = os.path.dirname(os.path.realpath(__file__))
#data_dir = os.path.abspath(os.path.join(this_dir, '../data/'))
raw_data_dir = '/scratch/pjohanss/arawling/collisionless-merger/'
processed_data_dir = '../../data/'
figure_dir = os.path.abspath(os.path.join(this_dir, '../figures/'))
figure_data_dir =  os.path.abspath(os.path.join(data_dir, 'figure_data/'))

#allow importing common analysis scripts/modules
#sys.path.append(os.path.abspath(os.path.join(this_dir, '../analysis_scripts')))

# get the complete path for saving a figure
def fig_path(fname):
    return os.path.join(figure_dir, fname)

# make sure we're using the same settings
matplotlib.rcdefaults()
matplotlib.rc_file(os.path.join(this_dir,'matplotlibrc'))
