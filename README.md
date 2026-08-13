<p align="center">
<img src="logo/baggins.png" width="300">
</p>

# BAGGInS: Bayesian Analysis of Galaxy-Galaxy Interactions in Simulations

## What is this repository for?  
* Initialisation, Bayesian-focused analysis, and presentation of 
collisionless merger simulations using the `KETJU` code developed by the 
Helsinki Theoretical Astrophyiscs Group.  
* Mostly the code is designed for use of *isolated* and *gas-free* mergers.

## How do I get set up?  
1. A few private repositories need to be installed and added to your python path - if you don't have access to these, contact the maintainer. The needed private repos are:  
- ketjugw  
- merger-ic-generator  
- pygad  
- voronoi-binning-cpp  
2. Next, install the BAGGInS package with `pip install -e .`  
3. Next, `cd baggins`, and copy a `env_params.yml` file to `baggins` and edit as necessary. The fields under the second document are internal settings of the code, and should not be edited.  

## Directory Organisation  
* all functions and classes can be found in `baggins/`.  
* Some basic scripts are located in `code/`.  

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

## Contributing  
Contributions are welcome. To keep the code base clean, we use `pre-commit` and `ruff`.  
```
pip install pre-commit
pip install ruff
pre-commit install
```
Now when you commit changes, automatic formatting and linting is done (in which case you'll need to rerun the `git add` and `git commit` commands).

### Who do I talk to?  
* Alex Rawlings (rawlings@mpa-garching.mpg.de)
