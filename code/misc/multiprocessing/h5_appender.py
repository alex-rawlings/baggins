import numpy as np
import cm_functions as cmf

f = "/scratch/pjohanss/arawling/testing/cube-H-H-0.05-0.02-000.hdf5"

cube = cmf.analysis.BHBinaryData.load_from_file(f)

new_data = np.arange(10)

cube.add_hdf5_field("new_data", new_data, "/some_new_data")