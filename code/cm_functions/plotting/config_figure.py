import os
import matplotlib

this_dir = os.path.dirname(os.path.realpath(__file__))
#set the matplotlib settings
matplotlib.rcdefaults()
matplotlib.rc_file(os.path.join(this_dir,'matplotlibrc'))