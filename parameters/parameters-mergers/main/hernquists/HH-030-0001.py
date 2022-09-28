#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "H"                       #galaxy 1 name
galaxyName2 = "H"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/galaxies/hernquist/hernquist.hdf5"
file2 = "/scratch/pjohanss/arawling/collisionless_merger/galaxies/hernquist/hernquist.hdf5"
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/mergers/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2-initialSeparation-pericentreDistance
perturbSubDir = "perturbations"


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1e-3"   #distance at first pericentre

#--- perturb properties
seed = 985541                       #seed for setting the RNG
perturbTime = 7.247                 #time when BH positions are perturbed
numberPerturbs = 10                 #how many perturbations to create
positionPerturb = 1.0e-2           #SD of position perturbation
velocityPerturb = 10              #SD of velocity perturbation
newParameterValues = {
                "SofteningStars":0.0035,
                "ketju_disable_integration": 0,
                "ErrTolIntAccuracy": 0.002
}                                   #parameter values to update, where keys are 
                                    #the name in the gadget paramfile 
                                    #"InitCondFile", "SnapshotFileBase" are 
                                    #always updated

#----------------------------------returned values
e = 9.99985e-01
full_save_location = "/scratch/pjohanss/arawling/collisionless_merger/mergers/H-H-3.0-0.001"
r0 = 5.17337e+02
rperi = 1.72446e-01
time_to_pericenter = 1.39707e+00
virial_radius = 1.72446e+02
#----------------------
