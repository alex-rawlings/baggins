import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad



snapfile = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0900/output")[-1]

snap = pygad.Snapshot(snapfile, physical=True)

xcom = pygad.analysis.shrinking_sphere(snap.stars, 
                                       pygad.analysis.center_of_mass(snap.stars),
                                       30)
trans = pygad.Translation(-xcom)
trans.apply(snap, total=True)

r = pygad.utils.geo.dist(snap["pos"])
v_sphere = bgs.mathematics.spherical_components(snap["pos"], snap["vel"])

r_edges = np.geomspace(1e-1, 20, 10)
bins_idxs = np.digitize(r, r_edges)

beta = []

rng = np.random.default_rng(42)

def beta_func(vs):
    sds = np.var(vs, axis=0)
    return 1 - (sds[1] + sds[2]) / (2 * sds[0])

'''for i in range(max(bins_idxs)):
    print(f"Doing bin {i} of {max(bins_idxs)}")
    mask = bins_idxs == i
    N = sum(mask)
    if N < 1e5:
        _beta = np.full(1000, np.nan)
        for j in range(1000):
            print(f"Resample {j}")
            _idxs = rng.choice(N, size=100)
            _vsphere = np.full((len(_idxs), 3), np.nan)
            for k, _idx in enumerate(_idxs):
                _vsphere[k,:] = v_sphere[mask,:][_idx,:]
            _beta[j] = beta_func(_vsphere)
        beta.append(np.mean(_beta))
    else:
        beta.append(beta_func(v_sphere[mask, :]))'''

rs = bgs.mathematics.get_histogram_bin_centres(r_edges)

#plt.plot(rs, beta, label="bootstrap")

beta_from_alex = bgs.analysis.velocity_anisotropy(snap, r_edges=r_edges, qcut=0.98)[0]

plt.loglog(rs, beta_from_alex, label="standard")

plt.legend()

plt.show()