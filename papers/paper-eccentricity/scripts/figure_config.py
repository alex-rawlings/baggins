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

def _make_color_cycle():
    """
    Construct the colour palette: a reordered, 10-element version of the 
    `twilight` colour map.
    """
    cols_f = lambda x: mpl.cm.twilight(mpl.colors.Normalize(-1,7)(x))
    #cols = [cols_f(i)  for p in zip(range(1,5), range(6,10)) for i in p]
    #cols += [cols_f(0), cols_f(5)]
    cols = cols_f([i for i in range(0,6,1)])

    return mpl.cycler(color=cols)

color_cycle = _make_color_cycle()
marker_cycle = mpl.cycler(marker=["o", "s", "^", "D", "v", "*", "p", "h", "X", "P"])

mpl.rcParams['axes.prop_cycle'] = color_cycle

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
