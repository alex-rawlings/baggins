# Description of the parameter file format for mergers

For reproducibility, the initial condition parameter values for each merger setup 
is saved within a parameter file from which the initial condition scripts reads 
the value in. Each parameter file consists of two sections: *specified inputs* 
and *returned values*. Unless otherwise stated:  
1. Masses are in solar masses  
2. distances are in kpc  
3. Time is in Gyr  
4. Velocity is in km/s  

Lines beginning with a `#` are comment lines, and ignored for the read in. The 
general structure is `parameter_name = parameter_value  #comment`. As the file 
is a .py file, it may be read in as one would a normal module.  
Herein *perturb* refers to the perturbation introduced to the BHs due to
Brownian motion.

## General  
- `galaxyName1`: name of the first galaxy in the merger setup  
- `galaxyName2`: name of the second galaxy in the merger setup  

## File Location  
- `file1`: isolated system (which presumably has been evolved to stability) 
for galaxy 1, should have been aligned and thus have `_aligned` in the file 
name.  
- `file2`: the same as `file1`, but for galaxy 2.  
- `saveLocation`: where the low mass resolution simulation will be saved to. 
  The format of the output directory is:  
        *name1-name2-r0-rperi*  
  where *name1* and *name2* are the progenitor names, and *r0* and *rperi* 
  are the initial separation and first pericentre distance, respectively.  
- `perturb1`: .pickle file containing data on the Brownian motion of the 
  isolated BH of galaxy 1. This file is output from 
  `bh_perturb_distributions.py`.  
- `perturb2`: the same as `perturb1`, but for galaxy 2.  
- `perturbSubDir`: the subdirectory (relative to `saveLocation`) where the set
  of perturbed runs will be saved.  

## Orbital Properties  
Note these properties are only used for the low mass realisation set up  
- `initialSeparation`: initial separation of the progenitors. Allowed inputs 
  are:  
    - `virialN`: the initial separation will *N* times the larger virial radius 
      of the two progenitors  
    - `touch`: no overlap of the progenitors (not implemented)  
    - `<number>`: an  arbitrary separation (not implemented)  
- `pericentreDistance`: separation at first pericentre passage. Allowed inputs 
  are:  
    - `virialN`: the initial separation will *N* times the larger virial radius 
      of the two progenitors  
    - `<number>`: an  arbitrary separation (not implemented)  

## Perturb Properties  
- `seed`: random seed for reproducibility  
- `perturbTime`: the time where to introduce the perturbations  
- `numberPerturbs`: the number of perturbed realisations to make  
- `positionPerturb`: the standard deviation in BH position due to Brownian 
  motion. This value is used for all BHs irrespective of host galaxy, and may be 
  found from the script `brownian_rotate_fit_normal.py`  
- `velocityPerturb`: the standard deviation in BH velocity due to Brownian 
  motion. This value is used for all BHs irrespective of host galaxy, and may be 
  found from the script `brownian_rotate_fit_normal.py`  
- `newParameterValues`: a python `dict` structure with `keyword:value` pairs, 
  where `keywords` are the parameter names to update in the `Gadget paramfile`, 
  and `value` is the value it should be updated to. Note that `InitCondFile` and 
  `SnapshotFileBase` are always updated.  


## Returned Values  
- `e`: orbital eccentricity of the merger  
- `full_save_location`: the full path of `saveLocation`  
- `r0`: numerical value of the initial separation  
- `rperi`: numerical value of the first pericentre passage separation  
- `time_to_pericenter`: time for the progenitors to reach the first pericentre 
  passage in the point-mass approximation  
- `virial_radius`: virial radius value used in the calculations

