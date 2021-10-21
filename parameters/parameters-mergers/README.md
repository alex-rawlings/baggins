# Description of the parameter file format for mergers

For reproducibility, the initial condition parameter values for each merger setup is saved within a parameter file from which the initial condition scripts reads the value in. Each parameter file consists of two sections: *specified inputs* and *returned values*. Unless otherwise stated:  
1. Masses are in solar masses  
2. distances are in kpc  
3. Time is in Gyr  
4. Velocity is in km/s  

Lines beginning with a `#` are comment lines, and ignored for the read in. The general structure is `parameter_name = parameter_value  #comment`. As the file is a .py file, it may be read in as one would a normal module.

## General  
- `regeneration`: (bool), is this merger simulation of the low -> high mass regeneration scheme?  
- `galaxyName1`: name of the first galaxy in the merger setup  
- `galaxyName2`: name of the second galaxy in the merger setup  

## File Location  
- `file1`: low mass resolution initial conditions for galaxy 1  
- `file2`: low mass resolution initial conditions for galaxy 2  
- `saveLocation`: where the low mass resolution simulation will be saved to. The format of the output directory is:  
        *name1-name2-r0-rperi*  
where *name1* and *name2* are the progenitor names, and *r0* and *rperi* are the initial separation and first pericentre distance, respectively  
- `fileHigh1`: the (stabilised) high mass resolution file to use as IC for the high mass resoluton realisation of galaxy 1. Will generally have the *aligned* keyword at the end of the file name.  
- `fileHigh2`: the (stabilised) high mass resolution file to use as IC for the high mass resoluton realisation of galaxy 2. Will generally have the *aligned* keyword at the end of the file name.  

## Orbital Properties  
Note these properties are only used for the low mass realisation set up  
- `initialSeparation`: initial separation of the progenitors. Allowed inputs are:  
    - `virialN`: the initial separation will *N* times the larger virial radius of the two progenitors  
    - `touch`: no overlap of the progenitors (not implemented)  
    - `<number>`: an  arbitrary separation (not implemented)
- `pericentreDistance`: separation at first pericentre passage. Allowed inputs are:  
    - `virialN`: the initial separation will *N* times the larger virial radius of the two progenitors  
    - `<number>`: an  arbitrary separation (not implemented)

## Returned Values  
- `e`: orbital eccentricity of the merger  
- `r0`: numerical value of the initial separation  
- `rperi`: numerical value of the first pericentre passage separation  
- `time_to_pericenter`: time for the progenitors to reach the first pericentre passage in the point-mass approximation  
- `virial_radius`: virial radius value used in the calculations