# parameter file for constructing Child Datacubes
import numpy as np

#############################
# file locations and saving #
#############################

# parent directory to save cubes to
cube_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes"



#############################
##### Binary properties #####
#############################

# radius above which GW emission should be negligible [pc]
gr_safe_radius = 15
# orbital parameters predicted for these e quantiles (must be of len 3)
param_estimate_e_quantiles = [0.05, 0.50, 0.95]



#############################
##### Galaxy properties #####
#############################

# stellar radius of galaxy, analagous to cosmo runs [pc]
galaxy_radius = 3e4
# dict of position separation and velocity separation of the two BHs below 
# which the CoM estimate is considered "converged"
com_consistency = {"pos":1e2, "vel":1.0}
# dict of tolerances below which the remnant is considered relaxed
relaxed_criteria = {"sep":0.01, "vrat":1.0}
# dict of values used for Voronoi tesselation, which is passed to 
# voronoi_binned_los_V_statistics()
voronoi_kw = {"Npx":300, "part_per_bin":5000}
#radius of a shell through which crossing statistics are computed [pc]
shell_radius = 30
# dict of sequence of values specifying the edge of the radial bins used for density profiles, beta profile [in pc]
radial_edges = {"stars": np.geomspace(20,2e4,51),
                "dm": np.geomspace(1e3,1e6,51),
                "all": np.geomspace(20, 1e6, 51)
                }
