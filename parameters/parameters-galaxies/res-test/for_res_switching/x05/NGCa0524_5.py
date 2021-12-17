#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 0.0                   #total simulation time
randomSeed = 265200                    #random seed
galaxyName = "A05"                #file name


#---- file information
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/"                            #parent save directory
figureLocation = "figures"        #figures saved in saveLocation/figureLocation
dataLocation = "output"           #simulation output saved here
litDataLocation = "literature_data"      #path to where literature data is


#---- stellar properties
stellarCored = 0                       #does the galaxy have a stellar core
logStellarMass = 11.4                  #total log10 stellar mass
stellarParticleMass = 5e5              #mass of stellar particle
stellarScaleRadius = 3.2               #star Dehnen scale radius
stellarGamma = 1.4                     #star Dehnen gamma
maximumRadius = 8000                   #maximum galaxy radius
minimumRadius = 1.5e-3                 #prevent tight orbits
anisotropyRadius = 0.4                 #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3             #stellar softening to be used


#---- halo properties
use_NFW = 1                              #use NFW profile, otherwise Dehnen
DMParticleMass = 1.5e8                     #mass of halo particle
DM_softening = 0.1                       #DM softening to be used
DM_mass_from = "Behroozi"                #which scaling relation to use
DM_cut = {"slope":4, "approx0":1e-5, "scaled_max_radius":7}  #cut parameters


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = "zlochower_dry"           #spin distribution to use
BH_spin = [-0.76785,+0.33621,-0.0172 ]  #BH chi vector
real_BH_mass = 8.94                      #measured log BH mass


#---- literature files
bulgeBHData = "sahu_20.txt"                #BH - bulge relation data
massData = "sdss_z.csv"                    #stellar mass distribution
BHsigmaData = "bosch_16.txt"               #BH - sigma relation data
fDMData = "jin_2020.dat"                   #inner dm fraction data


#----------------------------------returned values

BH_mass = 3.04220e+09
DM_peak_mass = 4.88851e+13
redshift = 0.00000e+00
#----------------------
DM_actual_total_mass = 9.65923e+13
DM_concentration = 7.54736e+00
count_BH = 1.00000e+00
count_DM_HALO = 6.43949e+05
count_STARS = 5.02056e+05
#----------------------
LOS_vel_dispersion = 2.72603e+02
half_mass_radius = 5.90176e+00
inner_1000_star_radius = 6.50670e-02
inner_100_star_radius = 1.53860e-02
inner_DM_fraction = 2.18746e-01
number_ketju_particles = 5.30000e+01
projected_half_mass_radius = 4.49883e+00
virial_mass = 3.95834e+13
virial_radius = 5.54290e+02
#----------------------
