import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import baggins as bgs

from sklearn.datasets import make_circles


data = make_circles(n_samples=1500, factor=0.2, noise=0.2)


default_cluster_params = {
    "quantile": 0.2,
    "eps": 0.3,
    "damping": 0.77,
    "preference": -240,
    "n_neghbors": 10,
    "n_clusters": 2,
    "min_samples": 20,
    "xi": 0.25,
    "min_cluster_size": 0.1
}

X, y = data

X = StandardScaler().fit_transform(X)

print(X)

spectral = cluster.SpectralClustering(
    n_clusters = default_cluster_params["n_clusters"],
    eigen_solver="arpack",
    affinity="nearest_neighbors"
)

spectral.fit(X)

y_pred = spectral.labels_.astype(int)
colors = np.array(bgs.plotting.mplColours())

print(y_pred)

plt.scatter(X[:,0], X[:,1], color=colors[y_pred])


plt.show()