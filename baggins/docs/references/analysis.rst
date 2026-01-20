analysis package
================

The analysis package includes several subpackages.
Notably, the main "themes" of the analysis package are direct calculations of quantities from snapshots and the higher-cadence `ketju_bhs.hdf5` file, interfacing to the MPA orbit analysis code, creation of mock IFU maps, and classes to facilitate Bayesian modelling of quantities - from density profiles to general-purpose Gaussian process regressors.

All methods and functions are called using::

   baggins.analysis.XXX

where `XXX` is the method name.

Modules
-------

.. toctree::
   :maxdepth: 2

   analysis.analyse_ketju
   analysis.analyse_snap
   analysis.instruments
   analysis.masks
   analysis.obs_helper
   analysis.orbits
   analysis.voronoi

Subpackages
-----------

.. toctree::
   :maxdepth: 2

   analysis.analysis_classes
   analysis.bayesian_classes
   analysis.data_classes
