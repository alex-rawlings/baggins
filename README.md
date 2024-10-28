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
* Mostly the code is designed for use of *isolated* and *gas-free* mergers, however there are plans to extend the functionality of BAGGInS to isolated hydro sims, BAGGInS+.  

## How do I get set up?  
1. A few private repositories need to be installed and added to your python path. These are:  
- ketjugw  
- merger-ic-generator  
- pygad  
- orbit-analysis  
- voronoi-binning-cpp  
2. Next, install the BAGGInS package with `pip install -e .`  
3. Next, `cd baggins`, and copy a `env_params.yml` file to `baggins` and edit as necessary. The fields under the second document are internal settings of the code, and should not be edited.  

## Directory Organisation  
* all script code required to perform the simulation initialisations and analysis is located in `code/`.  
* papers, including figure-specific scripts, can be found in `papers/`.  
* parameter files (for initialisation and analysis) for all simulations can be found in `parameters/`.  
These parameter files are expected to be used in conjunction with the 
functions and scripts in `baggins/` `code/`.

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

### Who do I talk to?  
* Alex Rawlings (alexander.rawlings@helsinki.fi)
