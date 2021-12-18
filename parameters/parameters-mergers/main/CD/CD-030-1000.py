#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "C"                       #galaxy 1 name
galaxyName2 = "D"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/NGCa3348/output/NGCa3348_055_aligned.hdf5"
file2 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/NGCa3607/output/NGCa3607_039_aligned.hdf5"
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/mergers/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2-initialSeparation-pericentreDistance
perturbSubDir = "perturbations"


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1e0"   #distance at first pericentre

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
e = 7.03114e-01
full_save_location = "/scratch/pjohanss/arawling/collisionless_merger/mergers/C-D-3.0-1.0"
r0 = 1.68594e+03
rperi = 5.61981e+02
time_to_pericenter = 3.22782e+00
virial_radius = 5.61981e+02
#----------------------
