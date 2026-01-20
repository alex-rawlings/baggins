.. baggins documentation master file, created by
   sphinx-quickstart on Thu Sep  7 10:23:14 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for BAGGInS
========================================

The Bayesian Analysis of Galaxy-Galaxy Interactions in Simulations (BAGGInS) tool is designed to facilitate a Bayesian-orientated approach to analysing galaxy merger simulations run with the Ketju extension of Gadget.
In addition to analysis of simulations, tools to set up different mergers, as well as handy plotting routines and utility functions, are included.
The tool has been primarily designed for isolated gas-free simulations, however a plan to extend the capability to work with isolated hydro-simulations, is in the works.

This document details all the functions, classes, and methods available to users as part of the BAGGInS toolset. 
There are eight main packages available to the user: `analysis`, `cosmology`, `general`, `initialise`, `literature`, `mathematics`, `plotting`, and `utils`.
These packages must be called explicitly to access their methods and classes, similar to how `scipy` works.
For example, to create a mock MUSE instrument observation, one would need to do::

   import baggins as bgs

   REDSHIFT = 0.2
   muse = bgs.analysis.MUSE_NFM(z=REDSHIFT)


and not::

   import baggins as bgs

   REDSHIFT = 0.2
   muse = bgs.MUSE_NFM(z=REDSHIFT)  # error

A user configurable parameter file `env_params.yml` is included to customise certain behaviours of the BAGGInS toolset. 
The form of the file is::

   ---
   cmdstan: # path to the cmdstan executable for Stan Monte Carlo sampling
   data_dir: # path to top level data/simulation directory
   date_format: '%Y-%m-%d %H:%M:%S'  # format for date strings
   figure_dir: # path to top level figure directory - figures are saved here
   figure_ext: .png  # figure format
   logging:
      console_level: INFO  # level of text output to screen
      file: logs/baggins.log  # file name to save logs to
      file_level: WARNING  # level of text output to log file
   mode: user  # mode of user operation
   synthesizer_data:  # directory to where synthesizer tables are located
   tmp_dir: tmp_dir  # prefix for temporary directories that are dynamically created and destroyed
   ...
   ---
   git_hash: heads/feature/docs-0-g6b9d503  # git hash of code version
   ...

Log files are created and stored for the previous ten days of usage before being overwritten.
This allows one to identify output after some analysis has been done, if the analysis has been done using e.g. a batch submission script (SLURM).

The `mode` parameter should in general be left as `user`: certain warnings and such that are relevant for active developers are turned off for clarity.

Finally, the `git_hash` parameter should be left untouched: this is used by BAGGInS as metadata for produced figures primarily.
In doing so, one may access the metadata of a generated figure to determine which code version was used to create it.

All questions should be directed to Alex Rawlings at: alexander.rawlings@helsinki.fi



.. toctree::
   :maxdepth: 1
   :caption: Available packages:
   :titlesonly:

   analysis <references/analysis>
   backend <references/_backend>
   cosmology <references/cosmology>
   general <references/general>
   initialise <references/initialise>
   literature <references/literature>
   mathematics <references/mathematics>
   plotting <references/plotting>
   utils <references/utils>



Indices and tables
==================

* :ref:`genindex`
