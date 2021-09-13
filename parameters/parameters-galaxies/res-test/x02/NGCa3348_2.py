#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 0.0                 #total simulation time
randomSeed = 985821                  #random seed
galaxyName = "C02"              #file name
distanceModulus = 33.08              #distance modulus to convert arcsec


#---- file information
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/02"                #parent save directory
figureLocation = "figures"      #figures saved in saveLocation/figureLocation
dataLocation = "output"         #simulation output saved here
litDataLocation = "literature_data"    #path to where literature data is


#---- stellar properties
stellarCored = 1                     #does the galaxy have a stellar core
sersicN = 3.8 #3.6                   #sersic index of bulge
effectiveRadius = 20.2               #effective radius in ARCSEC
logCoreDensity = 1.50 #1.70          #core density in MAG / ARCSEC^2
coreRadius = 0.59 #0.37              #core radius in ARCSEC
coreSlope = 0.58 #0.44               #slope profile of core region
transitionIndex = 7                  #how abruptly core changes to regular bulge
M2Lratio = 4                         #assumed mass-to-light ratio M_sol/L_sol
stellarParticleMass = 2e5            #mass of stellar particle
maximumRadius = 8000                 #maximum galaxy radius
minimumRadius = 1.5e-3               #prevent tight orbits
anisotropyRadius = 1.5               #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3           #stellar softening to be used


#---- halo properties
use_NFW = 1                          #use an NFW profile, otherwise Dehnen
DMParticleMass = 6.0e7               #mass of halo particle
DM_softening = 0.1                   #DM softening to be used
DM_mass_from = "Behroozi"            #which scaling relation to use
DM_cut = {"slope":4, "approx0":1e-5, "scaled_max_radius":7}  #cut parameters


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = "zlochower_dry"           #spin distribution to use
BH_spin = [+0.31454,+0.45651,+0.49157]  #BH chi vector

#---- literature files
bulgeBHData = "sahu_20.txt"                #BH - bulge relation data
massData = "sdss_z.csv"                    #stellar mass distribution
BHsigmaData = "bosch_16.txt"               #BH - sigma relation data
fDMData = "jin_2020.dat"                   #inner dm fraction data


#----------------------------------returned values
