# Things yet to be implemented in the cm_functions package

- [ ] Handling of literature data other than the defaults. This would require
  tailored reading in of data (maybe include a new submodule for this?)  
- [ ] Other separation and pericentre distances for merger_setup.py  
- [ ] Script to set up simulation data directory  
- [ ] Updating legacy numpy.random.seed() functions (in galaxy_gen.py), which will
  require updating the merger-ic-generator  
- [ ] Package dependencies available via requirements.txt, but not sure how pygad
  installed with pip will be affected?  
- [ ] Create a publishing .mplstyle format, to make plots more appealing for
  publishing  
- [ ] Refactoring initialisation scripts, functions, and classes, so the different 
  scripts are methods of a class  
- [ ] Converter for parameter files to .yml or .json for portability  
- [x] add metadata to figures produced  
- [x] define a top-level data directory so not so many /scratch/pjohanss/ stuff in parameter files  
- [x] check timestamp of copied keju_bhs file, copy only if necessary in get_ketjubhs_in_dir()
- [] implement logging statements instead of printing
- [] parallelise certain serial code using DASK 
    https://cosmiccoding.com.au/tutorials/multiprocessing