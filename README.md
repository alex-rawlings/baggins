# Collisionless Merger Sample #  

## What is this repository for?  
* Initialisation, analysis, and presentation of a large sample of 
collisionless merger simulations using the `KETJU` code developed by the 
Helsinki Theoretical Astrophyiscs Group  

## How do I get set up?  
1. Add the path to `./code/cm_functions` to the `PYTHONPATH` variable in 
your `~/.bashrc` profile to load the necessary functions as a normal 
module  

## Directory Organisation  
* all code (functions, scripts, classes, etc.) required to perform the 
simulation initialisations and analysis is located in `./code/`
* data (expected to be) from the output is saved in `./data/`, but is 
not uploaded to the BitBucket
* the paper(s) pertaining to this simulation set can be found in `./
paper/`, with each paper section as a separate file in `./paper/sections/`
* parameter files for all simulations can be found in `./parameters/`. 
These parameter files are expected to be used in conjunction with the 
functions and scripts in `./code/`

## Simulation Workflow  
The simulation workflow consists of a number of steps to produce 
realistic initial conditions for the merger. The steps are:  
1. Create both low and high mass resolution realisation of the 
progenitors using `create_select_galaxies.sh` in `./code/
initialise_scripts`
2. Evolve the high mass resolution progenitors in isolation until the 
system is stable (this is particularly relevant for Osipkov-Merritt 
models). The stability of the system can be checked with these scripts 
in `./code/analysis_scripts`:  
    - `inertia_analysis.py`, and
    - `beta_profile.py`  
3. Align the galaxy with the reduced inertia tensor semiminor axis using 
`align_galaxy.py` in `./code/analysis_scripts`
4. Concurrently, obtain the centre of mass motions of the system at 
large separations using the low mass realisations of the galaxies by 
running this merger as a standard simulation. Orbital configurations are 
given as inputs in the parameter file in `./parameters/
parameters-mergers/`.  
5. A high mass resolution system can then be generated as a combination 
of the low mass resolution centre of mass motions, and the stabilised 
high mass resolution galaxy, by running `extract_remake.py` for the 
corresponding parameter file in `./parameters/parameters-mergers/`.
6. The high mass resolution can then be started as if it were a new 
simulation. 
7. When to switch on Ketju???
8. Analyse key properties of the remnant system and SMBH dynamics using 
the provided scripts in `./code/analysis_scripts`.

## A Brief Note on the Scripts  
A detailed description of each initialisation and analysis script is not 
given here, however all scripts make use of the inbuilt `argparse` 
methods of python. If you are unsure how a script should be invoked, the 
best way to find out is by running  
`python ./script.py -h`  
A list of 
required inputs and optional inputs will be displayed.

### Contribution guidelines  
* Writing tests
* Code review
* Other guidelines

### Who do I talk to?  
* Alex Rawlings (alexander.rawlings@helsinki.fi)
* Matias Mannerkoski (matias.mannerkoski@helsinki.fi)
