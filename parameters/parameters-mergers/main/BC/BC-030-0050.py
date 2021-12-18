#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "B"                       #galaxy 1 name
galaxyName2 = "C"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/NGCa2986/output/NGCa2986_000_aligned.hdf5"
file2 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/NGCa3348/output/NGCa3348_055_aligned.hdf5"
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/mergers/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2-initialSeparation-pericentreDistance
perturbSubDir = "perturbations"


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial5e-2"   #distance at first pericentre

#--- perturb properties
seed = 985541                       #seed for setting the RNG
perturbTime = 99                 #time when BH positions are perturbed
numberPerturbs = 10                 #how many perturbations to create
positionPerturb = 1.11e-2           #SD of position perturbation
velocityPerturb = 12.8              #SD of velocity perturbation
newParameterValues = {
                "SofteningStars":0.0035,
                "ketju_disable_integration": 0,
                "ErrTolIntAccuracy": 0.002
}                                   #parameter values to update, where keys are 
                                    #the name in the gadget paramfile 
                                    #"InitCondFile", "SnapshotFileBase" are 
                                    #always updated

#----------------------------------returned values
e = 9.91681e-01
full_save_location = "/scratch/pjohanss/arawling/collisionless_merger/mergers/B-C-3.0-0.05"
r0 = 2.04296e+03
rperi = 3.40494e+01
time_to_pericenter = 1.87412e+00
virial_radius = 6.80988e+02
#----------------------
