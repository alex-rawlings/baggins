# Description of the parameter file format

For reproducibility, the initial condition parameter values for each model galaxy is saved within a parameter file from which the initial condition scripts reads the value in. Each parameter file consists of two sections: *specified inputs* and *returned values*. Unless otherwise stated:  
1. Masses are in solar masses  
2. distances are in kpc  
3. Time is in Gyr  
4. Velocity is in km/s  

Lines beginning with a `#` are comment lines, and ignored for the read in. The general structure is `parameter_name = parameter_value  #comment`. As the file is a .py file, it may be read in as one would a normal module.

A brief description of the *specified inputs* is given below. Note that depending on the model, not all parameter files below may be in the parameter file. If a parameter belongs to a specific model, that model abbreviation is given after the description. The model abbreviations are:  
1. Dehnen stellar profile: **SD**  
2. Cored stellar profile: **SC**  
3. Anisotropic Osipkov-Merrit model: **OM**  
4. Dehnen halo profile: **HD**  
5. NFW halo profile: **NFW**  

## General  
`simulationTime`: the (estimated) total simulation time -> used to estimate redshift-dependent properties of e.g. the halo mass  
`randomSeed`: number to seed the random number generator with, makes the generator deterministic  
`galaxyName`: prefix to attach to outputs  
`distanceModulus`: for unit conversion from arcsec to kpc (**SC**)  

## File Information  
`saveLocation`: parent directory where to save outputs to  
`figureLocation`: subdirectory to save figures to  
`dataLocation`: subdirectory where simulation data will be saved  
`litDataLocation`: directory (relative to where initialisation is run) where literature data is located  

## Stellar
`stellarCored`: bool, true if we want a Terzic-2005 cored density profile  
`sersicN`: sersic index (**SC**)  
`effectiveRadius`: half-light radius in arcsec (**SC**)  
`logCoreDensity`: density at the core radius in mag/arcsec^2 (**SC**)  
`coreRadius`: radius of the core in arcsec (**SC**)  
`coreSlope`: index of the core power law (**SC**)  
`transitionIndex`: rapidity of transition from core to sersic profile (**SC**)  
`M2Lratio`: assumed mass to light ratio in solar mass per solar luminosity (**SC**)  
`stellarMass`: log10 of total stellar mass (**SD**)  
`stellarScaleRadius`: scale radius (**SD**)  
`stellarGamma`: index of density profile (**SD**)  
`stellarParticleMass`: mass of a stellar particle  
`maximumRadius`: particles are not generated beyond this radius  
`minimumRadius`: particles are not generated within this radius  
`anisotropyRadius`: radius at which beta profiles becomes radially dominant (**OM**)  
`stellar_softening`: softening length of dm in the simulation  

## Halo  
`use_NFW`: bool, true if we want to use an NFW profile  
`DMParticleMass`: mass of a DM particle  
`DM_mass_from`: scaling relation from which to estimate halo mass from bulge mass  
`DM_softening`: softening length of stars in the simulation  
`DMScaleRadius`: scale radius (**HD**)  
`DMGamma`: index of density profile (**HD**)  

## SMBH  
`BH_softening`: softening length for SMBH particles  
`BH_spin`: dimensionless spin vector (random orientation), set as *random* to generate a new spin value, which overwrites the original *random* keyword  
`real_BH_mass`: actual SMBH mass if observed (not critical to pipeline)  

## Data Files  
`bulgeData1`: path to observational data of BH-bulge mass relation  
`massData`: path to observational data of ETG masses  
`BHsigmaData1`: path to observational data of BH-sigma relation  

## Returned Values  
`BH_mass`: SMBH mass estimate from bulge mass scaling relation  
`DM_peak_mass`: peak halo mass estimate from (specified) bulge mass scaling relation (refer to Behroozi+19) for definition  
`redshift`: estimated redshift from simulation time  
`input_Re_in_kpc`: effective radius in kpc (**SC**)  
`input_Rb_in_kpc`: core radius in kpc (**SC**)  
`DM_concentration`: concentration parameter (**NFW**)  
`DM_actual_total_mass`: the total DM mass generated  
`stellar_actual_total_mass`: the total stellar mass generated  
`STARS_count`: number of stellar particles in model  
`DM_HALO_count`: number of DM particles in model  
`number_ketju_particles`: estimated number of Ketju particles in the galaxy  
`BH_count`: number of SMBH particles in model (should be 1 generally)  
`halfMassRadius`: radius within which half the mass is located (**SD**)  
`projectedHalfMassRadius`: the projected half mass radius from 3 different viewing angles  
`innerDMfraction`: fraction of DM within projectedHalfMassRadius  
`virialRadius`: the measured virial radius  
`virialMass`: the measured virial mass  
`inner_100_star_radius`: radial displacement of the 100th stellar particle  
`inner_1000_star_radius`: radial displacement of the 1000th stellar particle  
`LOS_velocity_dispersion`: line of sight velocity dispersion within a half mass radius  
