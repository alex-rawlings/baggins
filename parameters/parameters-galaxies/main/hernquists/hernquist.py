#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 0.0                   #total simulation time
randomSeed = 4489822                    #random seed
galaxyName = "hernquist"                #file name


#---- file information
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/galaxies"                            #parent save directory
figureLocation = "figures"        #figures saved in saveLocation/figureLocation
dataLocation = "output"           #simulation output saved here
litDataLocation = "literature_data"      #path to where literature data is


#---- stellar properties
stellarCored = 0                       #does the galaxy have a stellar core
logStellarMass = 11.0                  #total log10 stellar mass
stellarParticleMass = 1e5              #mass of stellar particle
stellarScaleRadius = 4.0               #star Dehnen scale radius
stellarGamma = 1.0                     #star Dehnen gamma
maximumRadius = 8000                   #maximum galaxy radius
minimumRadius = 1.5e-3                 #prevent tight orbits
#anisotropyRadius = 0.4                 #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3             #stellar softening to be used


#---- halo properties
use_NFW = 0                              #use NFW profile, otherwise Dehnen
DMGamma = 1.0
DMScaleRadius = 300
DMParticleMass = 3e6                     #mass of halo particle
DM_softening = 0.1                       #DM softening to be used
DM_mass_from = "Behroozi"                #which scaling relation to use


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = "zlochower_dry"           #spin distribution to use
BH_spin = [+0.32915,-0.34561,+0.75191]  #BH chi vector


#----------------------------------returned values
BH_mass = 8.80457e+08
DM_peak_mass = 8.22069e+12
count_BH = 1.00000e+00
count_DM_HALO = 2.54572e+06
count_STARS = 9.99001e+05
redshift = 0.00000e+00
#----------------------
LOS_vel_dispersion = 1.05171e+02
half_mass_radius = +9.52436
inner_1000_star_radius = 1.34310e-01
inner_100_star_radius = 4.16538e-02
inner_DM_fraction = +0.09661
number_ketju_particles = 4.00000e+00
projected_half_mass_radius = 7.15852e+00
virial_mass = 1.19140e+12
virial_radius = 1.72418e+02
#----------------------
