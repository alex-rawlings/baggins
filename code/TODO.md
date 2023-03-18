# Things yet to be implemented in the cm_functions package

## Legend  
[ ]: yet to be implemented  
[x]: implemented  
[-]: decided against  

## Issues  

- [x] Handling of literature data other than the defaults. This would require
  tailored reading in of data (maybe include a new submodule for this?)  
- [x] Other separation and pericentre distances for merger_setup.py  
- [-] Script to set up simulation data directory  
- [x] Updating legacy numpy.random.seed() functions (in galaxy_gen.py), which will
  require updating the merger-ic-generator  
- [ ] Package dependencies available via requirements.txt, but not sure how pygad
  installed with pip will be affected?  
- [x] Create a publishing .mplstyle format, to make plots more appealing for
  publishing  
- [x] Refactoring initialisation scripts, functions, and classes, so the different 
  scripts are methods of a class  
- [x] Converter for parameter files to .yml or .json for portability  
- [x] add metadata to figures produced  
- [x] define a top-level data directory so not so many /scratch/pjohanss/ stuff in parameter files  
- [x] check timestamp of copied keju_bhs file, copy only if necessary in get_ketjubhs_in_dir()  
- [x] implement logging statements instead of printing  
- [x] parallelise certain serial code using DASK 
    https://cosmiccoding.com.au/tutorials/multiprocessing  
- [x] get_string_unique_part() doesn't seem to handle only two strings  
- [-] an analysis class A has a corresponding 'data class' AData, which may lead to the diamond problem when using multi-inheritance. Maybe combine the two classes, and methods in A which rely on variables not store in AData (i.e., not saved to HDF5 file) be restructured (typically to do with ketjugw orbital parameters -> maybe save the file name for future loading instead? What if the file has since been overwritten or updated?)
- [ ] apply perturbation as a total magnitude projected along different axes