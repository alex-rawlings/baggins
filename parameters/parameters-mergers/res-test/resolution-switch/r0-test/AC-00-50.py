#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
regeneration = True               #is this part of a regeneration run?
galaxyName1 = "A"                       #galaxy 1 name
galaxyName2 = "C"                       #galaxy 2 name


#--- file location
file1 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/A05/A05.hdf5"
file2 = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x05/C05/C05.hdf5"
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/r0-test/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2-initialSeparation-pericentreDistance

fileHigh1 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa0524/NGCa0524.hdf5"#output/NGCa0524_009.hdf5"            #file to regenerate IC from
fileHigh2 = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/NGCa3348/NGCa3348.hdf5"#output/NGCa3348_009.hdf5"             #file to regenerate IC from


#---- orbital properties
initialSeparation = "virial0.5"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1e-3"   #distance at first pericentre

#----------------------------------returned values
e = 9.99985e-01
r0 = 2.80981e+02
rperi = 5.61961e-01
time_to_pericenter = 2.24836e-01
virial_radius = 5.61961e+02
#----------------------
