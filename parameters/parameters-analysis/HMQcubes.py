# parameter file for constructing Hierarchical Model Quantity Datacubes
import numpy as np

#############################
# file locations and saving #
#############################

# parent directory to save cubes to
cube_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes"



#############################
##### Binary properties #####
#############################

# semimajor axis used to 'synchronise' simulations for fair comparison [kpc]
target_semimajor_axis = 10e-3



#############################
##### Galaxy properties #####
#############################

# stellar radius of galaxy, analagous to cosmo runs [kpc]
galaxy_radius = 30

# radial edges to be used for all radially binned quantities [kpc]
radial_edges = np.geomspace(0.2, galaxy_radius, 51)

# number of rotations to get projected quantity statistics
num_projection_rotations = 10



#############################
##### Stan Settings #####
#############################

# minimum number of allowed samples for hierarchical modelling
hm_min_num_samples = 6
# sampling kwargs
stan_sample_kwargs = {"adapt_delta":0.9}
