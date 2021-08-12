#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
simulationTime = 2.0                     #total simulation time
randomSeed = 265200                      #random seed
galaxyName = "test"                      #file name


#---- file information
saveLocation = "/Volumes/Rawlings_Storage/KETJU/initialise/"  #parent save directory
figureLocation = "figures"         #figures saved in saveLocation/figureLocation
dataLocation = "output"            #simulation output saved here
litDataLocation = "literature_data"    #path to where literature data is


#---- stellar properties
stellarCored = 0
logStellarMass = 11.4                    #total log10 stellar mass
stellarParticleMass = 1e5                #mass of stellar particle
stellarScaleRadius = 3.2                 #star Dehnen scale radius
stellarGamma = 1.4                       #star Dehnen gamma
maximumRadius = 8000                     #maximum galaxy radius
minimumRadius = 1.5e-3                   #prevent tight orbits
anisotropyRadius = 0.4                   #Osipkov Merrit anisotropy radius
stellar_softening = 3.5e-3               #stellar softening to be used


#---- halo properties
use_NFW = 1                              #use NFW profile, otherwise Dehnen
DMParticleMass = 6e7                     #mass of halo particle
DM_softening = 0.1                       #DM softening to be used
DM_mass_from = "Behroozi"                #which scaling relation to use


#---- black hole properties
BH_softening = 3e-3                      #BH softening length
BH_spin_from = 'zlochower_dry'           #spin distribution to use
BH_spin = [-0.76785,+0.33621,-0.0172 ]  #BH spin
real_BH_mass = 8.94                      #measured log BH mass


#---- literature files
bulgeBHData = "sahu_20.txt"           #BH - bulge relation data
massData = "sdss_z.csv"               #stellar mass distribution
BHsigmaData = "bosch_16.txt"          #BH - sigma relation data
fDMData = "jin_2020.dat"              #inner dm fraction data


#----------------------------------returned values
BH_mass = 3.04220e+09
DM_peak_mass = 6.02049e+13
redshift = 2.09136e-01
#----------------------
DM_actual_total_mass = 1.82272e+14
DM_concentration = 6.55445e+00
count_BH = 1.00000e+00
count_DM_HALO = 2.82766e+06
count_STARS = 2.51028e+06
#----------------------
LOS_vel_dispersion = 2.76069e+02
half_mass_radius = 5.90176e+00
inner_1000_star_radius = 2.52209e-02
inner_100_star_radius = 7.15401e-03
inner_DM_fraction = 2.44660e-01
number_ketju_particles = 2.23000e+02
projected_half_mass_radius = 4.39935e+00
virial_mass = 5.16422e+13
virial_radius = 6.05665e+02
#----------------------
