import os
import matplotlib as mpl
import numpy as np

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
mpl.rcdefaults()
mpl.rc_file(os.path.join(this_dir,'matplotlibrc_publish'))

custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                            'custom',
                            np.array([(30,46,118),
                                      (180,61,184),
                                      (201,49,49),
                                      (239,176,52)]
                                    )/256.
                            )
mpl.colormaps.register(cmap=custom_cmap) # can use as cmap='custom'

custom_colors = custom_cmap(np.linspace(0,1,6))
custom_colors_shuffled = custom_colors[[0,5,3,1,4,2,]]

color_cycle = mpl.cycler(color=custom_colors)
color_cycle_shuffled = mpl.cycler(color=custom_colors_shuffled)

marker_cycle = mpl.cycler(marker=["o", "s", "^", "D", "v", "*", "p", "h", "X", "P"])

mpl.rcParams['axes.prop_cycle'] = color_cycle_shuffled

marker_kwargs = {"edgecolor":"k", "lw":0.5}


class EccentricityScale(mpl.scale.FuncScale):
    """
    A non-linear scale for bound binary eccentricity plots.
    Approximately linear for values below ~0.6, then increasingly non-linear
    to expand the range between 0.9 and 1.
    """
    name = 'eccentricity'
    def __init__(self, axis):
        self.fac = 4.5 
        def forward(x):
            with np.errstate(divide="ignore", invalid="ignore"): 
                res = 1-10**(-np.arctanh(x)/self.fac)
                return np.where(x>1, 1000, res) # make invalid values be mapped outside the plot
        def inverse(x):
            with np.errstate(divide="ignore", invalid="ignore"): 
                return np.tanh(-self.fac*np.log10((1-x)))

        super().__init__(axis, (forward, inverse))

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mpl.ticker.FixedLocator([0,0.2,0.4,0.6,0.8,0.9, 0.99, 0.999, 1]))
        axis.set_minor_locator(mpl.ticker.FixedLocator([1-1e-4, 1-1e-5, 1-1e-6]))

        axis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:5g}"))

mpl.scale.register_scale(EccentricityScale)
