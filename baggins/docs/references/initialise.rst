initialise package
==================

The initialise package is designed to interface with the `merger_ic_generator` code developed at University of Helsinki to create initial conditions for galaxies and galaxy mergers using `.yml` parameter files. 
This allows for a general set-up script to be written, and then defining new initial conditions using different parameter files.

All methods and functions are called using::

   baggins.initialise.XXX

where `XXX` is the method name.

Modules
-------

initialise.GalaxyIC submodule
-----------------------------

.. automodule:: baggins.initialise.GalaxyIC
   :members:
   :undoc-members:
   :show-inheritance:

initialise.MergerIC submodule
-----------------------------

.. automodule:: baggins.initialise.MergerIC
   :members:
   :undoc-members:
   :show-inheritance:

initialise.galaxy\_components submodule
---------------------------------------
.. automodule:: baggins.initialise.galaxy_components
   :synopsis: Private classes that aren't instantiated directly, but listed here so one knows the options that are available
   :members: _GalaxyICBase, _StellarComponent, _StellarCusp, _StellarCore, _DMComponent, _DMHaloNFW, _DMHaloDehnen, _SMBH
   :private-members:
   :undoc-members:
   :show-inheritance:

initialise.mergers submodule
----------------------------

.. automodule:: baggins.initialise.mergers
   :members:
   :undoc-members:
   :show-inheritance:
