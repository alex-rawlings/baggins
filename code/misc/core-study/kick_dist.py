import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs


m1 = 1e-1
m2 = 1e-1
s1 = [0.,0.,-1.]
s2 = [0.,0.,1.]
x1 = [1e-3,0.001,0]
x2 = [-1e-3,0,0]
v1 = [1e-3,0.01,0]
v2 = [-1e-3,0,0]

rng = np.random.default_rng(42)
N = 5000

s1 = rng.uniform(0, 1, size=(N,3))
s1 /= np.linalg.norm(s1, axis=0)
s2 = rng.uniform(0, 1, size=(N,3))
s2 /= np.linalg.norm(s2, axis=0)

v = np.full(N, np.nan)

for i, (ss1, ss2) in enumerate(zip(s1, s2)):
    print(f"Sampling {(i+1)/N*100:.1f}% complete...                      ", end="\r")
    # convert unit of spin
    remnant = bgs.literature.ketju_calculate_bh_merger_remnant_properties(
        m1=m1, m2=m2,
        s1=ss1, s2=ss2,
        x1=x1, x2=x2,
        v1=v1, v2=v2
    )
    v[i] = np.linalg.norm(remnant["v"])
print("\nSampling complete")

v = np.sort(v)
P = 1-np.cumsum(v)/np.sum(v)
P = np.clip(P, 1e-7, None)

if True:
    plt.semilogy(v, P)
    plt.xlabel(r"$v_\mathrm{kick}$")
    plt.show()

if False:
    rb = lambda vv: 2.02 * (vv/1200)**0.42 + 1
    plt.hist(rb(v), 20, density=True)
    plt.axvline(rb(900), c="tab:red", lw=2)
    plt.ylabel("rb")
    plt.show()