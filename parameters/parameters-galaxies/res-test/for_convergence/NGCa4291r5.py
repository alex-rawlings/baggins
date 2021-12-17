#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 0.0                 #total simulation time
randomSeed = 5432                    #random seed
galaxyName = "E5"              #file name
distanceModulus = 32.09              #distance modulus to convert arcsec


#---- file information
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/resolution-convergence"                #parent save directory
figureLocation = "figures"      #figures saved in saveLocation/figureLocation
dataLocation = "output"         #simulation output saved here
litDataLocation = "literature_data"    #path to where literature data is


#---- stellar properties
stellarCored = 1                     #does the galaxy have a stellar core
sersicN = 5.6                        #sersic index of bulge
effectiveRadius = 21.8 #21.4         #effective radius in ARCSEC
logCoreDensity = 2.15 #2.30          #core density in MAG / ARCSEC^2
coreRadius = 0.55 #0.46              #core radius in ARCSEC
coreSlope = 0.52 #0.43               #slope profile of core region
transitionIndex = 7                  #how abruptly core changes to regular bulge
M2Lratio = 4                         #assumed mass-to-light ratio M_sol/L_sol
stellarParticleMass = 5e5            #mass of stellar particle
maximumRadius = 8000                 #maximum galaxy radius
minimumRadius = 1.5e-3               #prevent tight orbits
anisotropyRadius = 0.5               #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3           #stellar softening to be used


#---- halo properties
use_NFW = 1                          #use an NFW profile, otherwise Dehnen
DMParticleMass = 1.5e8               #mass of halo particle
DM_softening = 0.1                   #DM softening to be used
DM_mass_from = "Behroozi"            #which scaling relation to use
DM_cut = {"slope":4, "approx0":1e-5, "scaled_max_radius":7}  #cut parameters


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = "zlochower_dry"           #spin distribution to use
BH_spin = [-0.25426,-0.23164,+0.72249]  #BH chi vector
real_BH_mass = 8.99                      #measured log BH mass


#---- literature files
bulgeBHData = "sahu_20.txt"                #BH - bulge relation data
massData = "sdss_z.csv"                    #stellar mass distribution
BHsigmaData = "bosch_16.txt"               #BH - sigma relation data
fDMData = "jin_2020.dat"                   #inner dm fraction data


#----------------------------------returned values
BH_mass = 1.46460e+09
DM_peak_mass = 1.70362e+13
input_Rb_in_kpc = 6.98132e-02
input_Re_in_kpc = 2.76714e+00
redshift = 0.00000e+00
#----------------------
DM_actual_total_mass = 3.28608e+13
DM_concentration = 8.35991e+00
count_BH = 1.00000e+00
count_DM_HALO = 2.19072e+05
count_STARS = 2.91880e+05
stellar_actual_total_mass = 1.45940e+11
#----------------------
LOS_vel_dispersion = 2.58386e+02
inner_1000_star_radius = 5.02053e-02
inner_100_star_radius = 1.94306e-02
inner_DM_fraction = 1.28636e-01
number_ketju_particles = 1.70000e+01
projected_half_mass_radius = 2.82472e+00
virial_mass = 1.38449e+13
virial_radius = 3.90539e+02
#----------------------
