#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "Af"                       #galaxy 1 name
galaxyName2 = "Cf"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa0524/NGCa0524.hdf5"            #file to generate IC from
file2 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa3348/NGCa3348.hdf5"             #file to generate IC from
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/fiducial/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1"   #distance at first pericentre

#----------------------------------returned values
e = 7.03114e-01
r0 = 1.68714e+03
rperi = 5.62380e+02
virial_radius = 5.62380e+02
#----------------------
