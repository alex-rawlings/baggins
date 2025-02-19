import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

data = np.loadtxt("mag_data.txt")
cut_pixels = None

data[data > 20] = 20

if cut_pixels is not None:
    data = data[cut_pixels:-cut_pixels, cut_pixels:-cut_pixels]


bh_ridx = 43
bh_cidx = 55

#data[bh_ridx, bh_cidx] = 10

if True:
    kwargs = {"n_neighbors": 2, "contamination":1e-4, "leaf_size":10, "metric":"nan_euclidean"} # worked for 600 km/s snap 7
    #kwargs = {"n_neighbors": 20, "contamination":1e-3, "leaf_size":10, "metric":"nan_euclidean"}
    loc = LocalOutlierFactor(**kwargs)
elif False:
    loc = IsolationForest(contamination=5e-3, n_estimators=100, bootstrap=True)
else:
    loc = make_pipeline(
        Nystroem(random_state=42, gamma=5e-3),
        SGDOneClassSVM(
            nu = 1e-1, 
            shuffle=True,
            fit_intercept=True,
            random_state=42,
            tol=1e-6
        )
    )

X = np.array([[i, j, data[i, j]] for i in range(data.shape[0]) for j in range(data.shape[1])])
predictions = loc.fit_predict(X)
outlier_mask = (predictions == -1)

print(f"There are {np.sum(outlier_mask)} outliers")

outlier_grid = np.zeros_like(data)
for index, is_outlier in enumerate(outlier_mask):
    i, j, _ = X[index]
    if is_outlier:
        outlier_grid[int(i), int(j)] = 1  # Mark outliers

im = plt.imshow(data, cmap='cividis', interpolation='nearest')
plt.colorbar(im, ax=plt.gca())
plt.imshow(outlier_grid, cmap='Reds', alpha=0.8)  # Overlay outliers in red
#plt.scatter(bh_ridx, bh_cidx, ec="green", fc=None, s=100, lw=1)
plt.savefig("outliers.png", dpi=300)