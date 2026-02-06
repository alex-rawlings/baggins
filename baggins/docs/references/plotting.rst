plotting package
================

The plotting package includes routines to create specific plots, as well as helper functions that can be used with any `matplotlib.axes.Axes` object - like creating normed colour scales, adding a scale bar to a figure, or adding an arrow to a curve.

The package includes custom colour maps for use in IFU mock images, and the BAGGInS-wide `matplotlibrc` file (which controls the default settings for generated figures) lives here.

An important method in this package is::

   bgs.plotting.savefig()

Which wraps around matplotlib's `savefig()` function, but adds extra metadata to the figure.

All methods and functions are called using::

   baggins.plotting.XXX

where `XXX` is the method name.

Submodules
----------
.. toctree::
   :maxdepth: 2

   plotting.PlotClasses
   plotting.custom_cmaps
   plotting.general
   plotting.utils
   plotting.specific_plots



Other
-----
Custom colour maps are available for Voronoi plots
