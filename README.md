# Collisionless Merger Sample  

## What is this repository for?  
* Initialisation, analysis, and presentation of a large sample of 
collisionless merger simulations using the `KETJU` code developed by the 
Helsinki Theoretical Astrophyiscs Group  

## How do I get set up?  
1. Add the path to `./code/cm_functions` to the `PYTHONPATH` variable in 
your `~/.bashrc` profile to load the necessary functions as a normal 
module  
2. The required python packages are given in the file `./code/requirements.txt`.
To install the required packages: `python -m pip install -r requirements.txt`. 
Note that `pygad` will probably need to be installed separately: easiest way is to clone it from the group BitBucket page and 
```
cd pygad
pip install . -e
```
as described in the `pygad` docs.
Also need to install `ketjugw`, `voronoi-binning-cpp`, `merger-ic-generator` from the  group BitBucket, and add these modules to your `$PYTHONPATH` in the `~/.bashrc` file.  
3. Edit the necessary fields in `./code/cm_functions/env_params.json` that are listed under `user_settings` (for e.g. directory where to save the figures to). The fields under `internal_settings` should not be edited by the user directly.  

## Directory Organisation  
* all code (functions, scripts, classes, etc.) required to perform the 
simulation initialisations and analysis is located in `./code/`
* the paper(s) pertaining to this simulation set can be found in `./
paper/`, with each paper section as a separate file in `./paper/sections/`
* parameter files for all simulations can be found in `./parameters/`. 
These parameter files are expected to be used in conjunction with the 
functions and scripts in `./code/`

## Simulation Workflow  
The simulation workflow consists of a number of steps to produce 
realistic initial conditions for the merger. The steps are:  
1. Create the progenitor galaxies using `make_galay.py` in 
`./code/initialise_scripts`. A valid parameter file must be first created.  
2. Evolve the progenitors in isolation until the 
system is stable (this is particularly relevant for Osipkov-Merritt 
models). The stability of the system can be checked with these scripts 
in `./code/analysis_scripts`:  
    - `inertia_analysis.py`, and  
    - `beta_profile.py`  
3. Align the galaxy with the reduced inertia tensor semiminor axis using 
`align_galaxy.py` in `./code/analysis_scripts`.  
4. Set up the merger configuration with a separate merger parameter file, using
the script `merger_setup.py`.  
5. Run the merger configuration using `Gadget` with reduced integration accuracy
(for this project, default $\eta=0.02$). Run the system until the BHs are 
bound.  
6. The Brownian motion of the BH in the isolated galaxy can be investigated
using `bh_perturb_distribution.py` and `brownian_rotate_fit_normal.py` in 
`./code/analysis_scripts`. Ensure that only those times where the galaxy is
relaxed is included in the latter analysis, by limiting the distribution fits
to a time `T` after the system is stable with the `-t T` flag.  
7. Identify the time when the BHs form a bound binary using 
`check_merger_progress.py <file> -b` in the directory `./code/analysis_scripts`.
This will produce a plot showing the radial separation of the binaries, as well
as the orbital energy of the binary system. Those points where the binary 
energy transitions from $\mathcal{E}>0$ to $\mathcal{E}<0$ are indicated by
numbered arrows, and the corresponding times printed to the console. The desired
time to turn on `Ketju` can thus be found.  
8. Create the children perturbed runs of the main `Gadget` run using the script
`perturb_bhs.py` in `./code/initialise_scripts` with the merger parameter file.
This generates a set of runs with the phase-space coordinates of the BHs 
randomised using the Brownian motion distributions of step 6. Other `Gadget
paramfile` options may be changed here too (especially $\eta=0.002$).  
9. Run each pertured run with `Ketju`. The analytical time to merger can be 
estimated using `hardening_rates.py` in `./code/analysis_scripts`. If the BHs
have not merged within the upper limit set by the analytical expectation, the
run may need to be terminated.  
10. Datacubes of the key quantities for both the SMBH binary, and the galaxy properties, can be created to facilitate later analysis.
This done using `make_child_datacubes.py` in `codes/analysis_scripts`.  

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
