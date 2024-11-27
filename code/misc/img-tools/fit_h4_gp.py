import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import baggins as bgs

from sklearn.datasets import make_circles


# set up the classifier
classifier = GaussianProcessClassifier(1.0 * kernels.RBF(1.0), random_state=42)

# load the data
data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/IFU_bt/ifu_bt_0600.pickle"
data = bgs.utils.load_data(data_file)
img = data["voronoi_stats_box"]["img_h4"]


fig, ax = plt.subplots()


c = make_circles(noise=0.2, factor=0.5, random_state=1)

print(c)