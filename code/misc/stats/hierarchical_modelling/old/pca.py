import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cm_functions as cmf
from ketjugw.units import yr

"""
num_subdirs = 10
subdirs = [f"{i:03d}" for i in range(num_subdirs)]

X = []

for s in subdirs:
    data = cmf.utils.load_data(f"../merger-test/pickle/early-{s}.pickle")
    X.append(data["infl_rad"]["bh2"])
X = np.array(X)
print(X.shape)"""


def get_and_centre_bhs(k, bound=False):
    if bound:
        bh1, bh2, merged = cmf.analysis.get_bound_binary(k)
    else:
        bh1, bh2, merged = cmf.analysis.get_bh_particles(k)
    mass_sum = np.atleast_2d(bh1.m + bh2.m).T
    xcom = (np.atleast_2d(bh1.m).T * bh1.x + np.atleast_2d(bh2.m).T * bh2.x) / mass_sum
    vcom = (np.atleast_2d(bh1.m).T * bh1.v + np.atleast_2d(bh2.m).T * bh2.v) / mass_sum
    vcom
    bh1.x -= xcom
    bh1.v -= vcom
    bh2.x -= xcom
    bh2.v -= vcom

    return bh1, bh2

X = []
minlen = 9e9
#maindir = "/scratch/pjohanss/arawling/collisionless_merger/high-time-output/A-C-3.0-0.05-H/"
maindir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/"
for kf in cmf.utils.get_ketjubhs_in_dir(maindir):
    bh1, bh2 = get_and_centre_bhs(kf, bound=True)
    mask = bh1.t/yr < 1e9
    L = cmf.mathematics.radial_separation(np.cross(bh1.x[mask], bh1.v[mask]))
    print(L.shape)
    X.append(L)
    if len(bh1[mask]) < minlen: minlen = len(bh1[mask])
for i in range(10):
    X[i] = X[i][:minlen]
X = np.array(X)
print(X.shape)



pca = PCA(n_components=10)
Xnew = pca.fit_transform(X)
print(Xnew.shape)
for i in range(10):
    plt.scatter(Xnew[i,0], Xnew[i,1])
plt.show()