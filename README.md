```
888888b.         d8888  .d8888b.   .d8888b.  8888888           .d8888b.   
888  "88b       d88888 d88P  Y88b d88P  Y88b   888            d88P  Y88b  
888  .88P      d88P888 888    888 888    888   888            Y88b.       
8888888K.     d88P 888 888        888          888   88888b.   "Y888b.    
888  "Y88b   d88P  888 888  88888 888  88888   888   888 "88b     "Y88b.  
888    888  d88P   888 888    888 888    888   888   888  888       "888  
888   d88P d8888888888 Y88b  d88P Y88b  d88P   888   888  888 Y88b  d88P  
8888888P" d88P     888  "Y8888P88  "Y8888P88 8888888 888  888  "Y8888P"   
                                                                          
```

# BAGGInS: Bayesian Analysis of Galaxy-Galaxy Interactions in Simulations

## What is this repository for?  
* Initialisation, Bayesian-focused analysis, and presentation of 
collisionless merger simulations using the `KETJU` code developed by the 
Helsinki Theoretical Astrophyiscs Group.  

## How do I get set up?  
1. Add the path to `code/baggins` to the `PYTHONPATH` variable in 
your `~/.bashrc` profile to load the necessary functions as a normal 
module  
2. The required python packages are given in the file `code/requirements.txt`.
To install the required packages: `python -m pip install -r requirements.txt`. 
Note that `pygad` will probably need to be installed separately: easiest way is to clone it from the group BitBucket page and 
```
cd pygad
pip install . -e
```
as described in the `pygad` docs.
Also need to install `ketjugw`, `voronoi-binning-cpp`, `merger-ic-generator`, `orbit-analysis` from the  group BitBucket, and add these modules to your `$PYTHONPATH` in the `~/.bashrc` file.  
3. Edit the necessary fields in `code/baggins/env_params.yml` that are listed under the top document (for e.g. directory where to save the figures to). The fields under the second document are internal settings of the code, and should not be edited.  
4. You'll need to compile the C code for some optimised routines. Make sure that you have given GitHub your public SSH key (for the device you're compiling on). Note that on Mahti, the module `git` doesn't seem to work with these commands; the base `git` does. As well, on Mahti, unload any `python-data` module during compilation so that the root python can be used.  
```
git submodule update --init
cd code/baggins/C
mkdir build && cd build
cmake ..
make
```

## Directory Organisation  
* all code (functions, scripts, classes, etc.) required to perform the 
simulation initialisations and analysis is located in `code/`.  
* papers, including figure-specific scripts, can be found in `papers/`.  
* parameter files (for initialisation and analysis) for all simulations can be found in `parameters/`.  
These parameter files are expected to be used in conjunction with the 
functions and scripts in `code/`.

## A Brief Note on the Scripts  
A detailed description of each initialisation and analysis script is not 
given here, however all scripts make use of the inbuilt `argparse` 
methods of python. If you are unsure how a script should be invoked, the 
best way to find out is by running  
```
python ./script.py -h
```
A list of 
required inputs and optional inputs will be displayed.  

## Developer Notes  
### Adding the pybind module  
To add the pybind module (this should not need to be done when generally cloning the repo), first do:  
```
git submodule add -b stable git@github.com:pybind/pybind11.git code/baggins/C/thirdparty/pybind11
```
before running the `git submodule update --init` steps.  

### Who do I talk to?  
* Alex Rawlings (alexander.rawlings@helsinki.fi)
* Matias Mannerkoski (matias.mannerkoski@helsinki.fi)
