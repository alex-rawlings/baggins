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
file1 = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/hernquist.hdf5"
file2 = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/hernquist.hdf5"
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2-initialSeparation-pericentreDistance
perturbSubDir = "perturbations-gadget"


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial5e-2"   #distance at first pericentre

#--- perturb properties
seed = 985541                       #seed for setting the RNG
perturbTime = 4.719                 #time when BH positions are perturbed
numberPerturbs = 10                 #how many perturbations to create
positionPerturb = 1.11e-2           #SD of position perturbation
velocityPerturb = 12.8              #SD of velocity perturbation
newParameterValues = {
                "ErrTolIntAccuracy": 0.002,
                "TimeMax": 4.0
}                                   #parameter values to update, where keys are 
                                    #the name in the gadget paramfile 
                                    #"InitCondFile", "SnapshotFileBase" are 
                                    #always updated

#----------------------------------returned values
e = 9.91681e-01
full_save_location = "/scratch/pjohanss/arawling/collisionless_merger/hernquist/H-H-3.0-0.05"
r0 = 9.25331e+02
rperi = 1.54222e+01
time_to_pericenter = 1.71151e+00
virial_radius = 3.08444e+02
#----------------------
perturb_snap_idx = 4.80000e+01
#----------------------
