#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "A20"                       #galaxy 1 name
galaxyName2 = "C20"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x20/A20/A20.hdf5"            #file to generate IC from
file2 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x20/C20/C20.hdf5"             #file to generate IC from
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x20/1-000/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1"   #distance at first pericentre

#----------------------------------returned values
e = 7.03114e-01
r0 = 1.68814e+03
rperi = 5.62715e+02
virial_radius = 5.62715e+02
#----------------------
