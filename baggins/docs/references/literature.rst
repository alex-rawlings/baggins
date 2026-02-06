literature package
==================

The literature package encompasses select observational data from other studies that may be accessed through the `LiteratureTables` class.
For example, to access data of dynamically hot stellar systems from Misgeld et al. 2011 and quickly make a scatter plot, one would do::

   import baggins as bgs
   lt = bgs.literature.load_misgeld_2011_data()
   lt.scatter("mass", "Re_pc")

Functional relations are also included here, such as the Moster et al. 2013 relation between stellar and halo mass.

All methods and functions are called using::

   baggins.literature.XXX

where `XXX` is the method name.

Modules
-------

.. toctree::
   :maxdepth: 2

   literature.LiteratureTables
   literature.bh_bulge
   literature.density_profiles
   literature.dm_bulge
   literature.radial_relations
   literature.simulation_fits
   literature.smbh_recoil
   literature.smbh_spins