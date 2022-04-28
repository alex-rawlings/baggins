#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "test1"                       #galaxy 1 name
galaxyName2 = "test1"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/NGCa3607/output/NGCa3607_039_aligned.hdf5"
file2 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/NGCa4291/output/NGCa4291_027_aligned.hdf5"
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/merger-test/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2-initialSeparation-pericentreDistance


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1e-3"   #distance at first pericentre

#--- perturb properties
perturbTime = 5.256                 #time when BH positions are perturbed
numberPerturbs = 10                 #how many perturbations to create

#----------------------------------returned values
e = 9.99985e-01
full_save_location = "/scratch/pjohanss/arawling/collisionless_merger/merger-test/test1-test1-3.0-0.001"
r0 = 1.23663e+03
rperi = 4.12212e-01
time_to_pericenter = 1.58646e+00
virial_radius = 4.12212e+02
#----------------------
