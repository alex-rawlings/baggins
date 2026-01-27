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

Submodules
----------

literature.LiteratureTables submodule
-------------------------------------

.. automodule:: baggins.literature.LiteratureTables
   :members:
   :undoc-members:
   :show-inheritance:

literature.bh\_bulge submodule
------------------------------

.. automodule:: baggins.literature.bh_bulge
   :members:
   :undoc-members:
   :show-inheritance:

literature.density\_profiles submodule
---------------------------------------

.. automodule:: baggins.literature.density_profiles
   :members:
   :undoc-members:
   :show-inheritance:

literature.dm\_bulge submodule
------------------------------

.. automodule:: baggins.literature.dm_bulge
   :members:
   :undoc-members:
   :show-inheritance:

literature.radial\_relations submodule
--------------------------------------

.. automodule:: baggins.literature.radial_relations
   :members:
   :undoc-members:
   :show-inheritance:

literature.smbh\_recoil submodule
---------------------------------

.. automodule:: baggins.literature.smbh_recoil
   :members:
   :undoc-members:
   :show-inheritance:

literature.smbh\_spins submodule
--------------------------------

.. automodule:: baggins.literature.smbh_spins
   :members:
   :undoc-members:
   :show-inheritance:
