#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 0.0                   #total simulation time
randomSeed = 4449822                    #random seed
galaxyName = "hernquist"                #file name


#---- file information
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger"                            #parent save directory
figureLocation = "figures"        #figures saved in saveLocation/figureLocation
dataLocation = "output"           #simulation output saved here
litDataLocation = "literature_data"      #path to where literature data is


#---- stellar properties
stellarCored = 0                       #does the galaxy have a stellar core
logStellarMass = 11.0                  #total log10 stellar mass
stellarParticleMass = 1e5              #mass of stellar particle
stellarScaleRadius = 5.0               #star Dehnen scale radius
stellarGamma = 1.0                     #star Dehnen gamma
maximumRadius = 8000                   #maximum galaxy radius
minimumRadius = 1.5e-3                 #prevent tight orbits
#anisotropyRadius = 0.4                 #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3             #stellar softening to be used


#---- halo properties
use_NFW = 1                              #use NFW profile, otherwise Dehnen
DMParticleMass = 3e7                     #mass of halo particle
DM_softening = 0.1                       #DM softening to be used
DM_mass_from = "Behroozi"                #which scaling relation to use
DM_cut = {"slope":4, "approx0":1e-5, "scaled_max_radius":7}  #cut parameters


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = "zlochower_dry"           #spin distribution to use
BH_spin = "random"  #BH chi vector
#real_BH_mass = 8.94                      #measured log BH mass


#---- literature files
bulgeBHData = "sahu_20.txt"                #BH - bulge relation data
massData = "sdss_z.csv"                    #stellar mass distribution
BHsigmaData = "bosch_16.txt"               #BH - sigma relation data
fDMData = "jin_2020.dat"                   #inner dm fraction data


#----------------------------------returned values

BH_mass = 8.80457e+08
DM_peak_mass = 8.22069e+12
redshift = 0.00000e+00
#----------------------
DM_actual_total_mass = 1.56058e+13
DM_concentration = 8.97220e+00
count_BH = 1.00000e+00
count_DM_HALO = 5.20195e+05
count_STARS = 9.98751e+05
#----------------------
LOS_vel_dispersion = 1.43560e+02
half_mass_radius = 1.20711e+01
inner_1000_star_radius = 1.60688e-01
inner_100_star_radius = 4.95787e-02
inner_DM_fraction = 6.60536e-01
number_ketju_particles = 3.00000e+00
projected_half_mass_radius = 8.93844e+00
virial_mass = 6.82062e+12
virial_radius = 3.08441e+02
#----------------------
