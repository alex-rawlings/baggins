#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "A10"                       #galaxy 1 name
galaxyName2 = "C10"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x10/A10/A10.hdf5"            #file to generate IC from
file2 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x10/C10/C10.hdf5"             #file to generate IC from
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x10/0-001/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1e-3"   #distance at first pericentre

#----------------------------------returned values
e = 9.99985e-01
r0 = 1.68807e+03
rperi = 5.62689e-01
virial_radius = 5.62689e+02
#----------------------
