import numpy as np
import baggins as bgs

f = "/scratch/pjohanss/arawling/testing/cube-H-H-0.05-0.02-000.hdf5"

cube = bgs.analysis.BHBinaryData.load_from_file(f)

new_data = np.arange(10)

cube.add_hdf5_field("new_data", new_data, "/some_new_data")