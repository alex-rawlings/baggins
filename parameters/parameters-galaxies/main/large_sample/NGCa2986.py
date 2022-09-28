#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 0.0                 #total simulation time
randomSeed = 34737                   #random seed
galaxyName = "NGCa2986"              #file name
distanceModulus = 32.31              #distance modulus to convert arcsec


#---- file information
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests"                  #parent save directory
figureLocation = "figures"        #figures saved in saveLocation/figureLocation
dataLocation = "output"           #simulation output saved here
litDataLocation = "literature_data"      #path to where literature data is


#---- stellar properties
stellarCored = 1                     #does the galaxy have a stellar core
sersicN = 7.0                        #sersic index of bulge
effectiveRadius = 70.1               #effective radius in ARCSEC
logCoreDensity = 1.8                 #core density in MAG / ARCSEC^2
coreRadius = 0.72                    #core radius in ARCSEC
coreSlope = 0.8                      #slope profile of core region
transitionIndex = 7                  #how abruptly core changes to regular bulge
M2Lratio = 4                         #assumed mass-to-light ratio M_sol/L_sol
stellarParticleMass = 1e5            #mass of stellar particle
maximumRadius = 8000                 #maximum galaxy radius
minimumRadius = 1e-3                 #prevent tight orbits
anisotropyRadius = 5                 #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3           #stellar softening to be used


#---- halo properties
use_NFW = 1                          #use an NFW profile, otherwise Dehnen
DMParticleMass = 3.0e7               #mass of halo particle
DM_softening = 0.1                   #DM softening to be used
DM_mass_from = "Behroozi"            #which scaling relation to use
DM_cut = {"slope":4, "approx0":1e-5, "scaled_max_radius":7}  #cut parameters


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = "zlochower_dry"           #spin distribution to use
BH_spin = [-0.23178,+0.14089,-0.68102]  #BH chi vector


#---- literature files
bulgeBHData = "sahu_20.txt"                #BH - bulge relation data
massData = "sdss_z.csv"                    #stellar mass distribution
BHsigmaData = "bosch_16.txt"               #BH - sigma relation data
fDMData = "jin_2020.dat"                   #inner dm fraction data


#----------------------------------returned values
BH_mass = 4.67899e+09
DM_peak_mass = 9.18642e+13
input_Rb_in_kpc = 1.01136e-01
input_Re_in_kpc = 9.84675e+00
redshift = 0.00000e+00
#----------------------
DM_actual_total_mass = 1.84259e+14
DM_concentration = 7.09937e+00
count_BH = 1.00000e+00
count_DM_HALO = 6.14198e+06
count_STARS = 3.45846e+06
stellar_actual_total_mass = 3.45854e+11
#----------------------
LOS_vel_dispersion = 2.99430e+02
inner_1000_star_radius = 2.77878e-02
inner_100_star_radius = 9.43645e-03
inner_DM_fraction = 5.49202e-01
number_ketju_particles = 1.24000e+02
projected_half_mass_radius = 9.89523e+00
virial_mass = 7.34054e+13
virial_radius = 6.80988e+02
#----------------------
