import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf


GP = cmf.analysis.StanModel_2D("gp.stan", "", "gp")

theta = []
e_hard = []

for f in cmf.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_250K-D_250K-3.720-0.028"):
    hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
    theta.append(
        cmf.analysis.first_major_deflection_angle(hmq.prebound_deflection_angles)
    )
    e_hard.append(hmq.ec)
