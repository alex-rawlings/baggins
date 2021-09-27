#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "A05"                       #galaxy 1 name
galaxyName2 = "C05"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/A05/A05.hdf5"            #file to generate IC from
file2 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/C05/C05.hdf5"             #file to generate IC from
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/0-180/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1.8e-1"   #distance at first pericentre

#----------------------------------returned values
e = 9.43488e-01
r0 = 1.68588e+03
rperi = 1.01153e+02
virial_radius = 5.61961e+02
#----------------------
