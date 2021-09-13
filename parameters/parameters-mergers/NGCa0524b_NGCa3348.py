#----------------------------------user input

#always assume units of:
#mass: solar masses
#distance: kpc
#time: Gyr
#velocity: km/s
#unless otherwise stated


#---- general
galaxyName1 = "A"                       #galaxy 1 name
galaxyName2 = "C"                       #galaxy 2 name


#--- file location
#file1 = "/Volumes/Rawlings_Storage/KETJU/initialise/galaxies/NGCa0524b/NGCa0524b.hdf5"
#file2 = "/Volumes/Rawlings_Storage/KETJU/initialise/galaxies/NGCa3348/NGCa3348.hdf5"
#saveLocation = "/Volumes/Rawlings_Storage/KETJU/initialise/merger/"

file1 = "/scratch/pjohanss/arawling/collisionless_merger/stability-test/NGCa0524b/output/NGCa0524_008.hdf5"            #file to generate IC from
file2 = "/scratch/pjohanss/arawling/collisionless_merger/stability-test/NGCa3348/output/NGCa3348_009.hdf5"             #file to generate IC from
saveLocation = "/scratch/pjohanss/arawling/collisionless_merger/res-test/fiducial/"      #file will be saved to saveLocation/galaxyName_1-galaxyName_2


#---- orbital properties
initialSeparation = "virial3"       #'touch', 'overlapXX', 'virialX' or number
pericentreDistance = "virial1e-3"   #distance at first pericentre

#----------------------------------returned values
e = 9.99985e-01
r0 = 1.20901e+03
rperi = 6.04506e-01
#----------------------
virial_radius = 6.04506e+02
#----------------------
