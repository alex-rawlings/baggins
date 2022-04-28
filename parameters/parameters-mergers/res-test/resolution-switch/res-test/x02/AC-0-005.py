#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "A02"                       #galaxy 1 name
galaxyName2 = "C02"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x02/A02/A02.hdf5"            #file to generate IC from
file2 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x02/C02/C02.hdf5"             #file to generate IC from
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x02/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial5e-3"   #distance at first pericentre

#----------------------------------returned values
e = 9.99799e-01
r0 = 1.68705e+03
rperi = 2.81176e+00
virial_radius = 5.62351e+02
#----------------------
