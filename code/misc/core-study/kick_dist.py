import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import baggins as bgs
from ketjugw.units import km_per_s


kfile = bgs.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output")[0]
bh1, bh2, *_ = bgs.analysis.get_bound_binary(kfile)
# move to Gadget units: kpc, km/s, 1e10Msol
bh1.x /= bgs.general.units.kpc
bh2.x /= bgs.general.units.kpc
bh1.v /= km_per_s
bh2.v /= km_per_s
bh1.m /= 1e10
bh2.m /= 1e10

rng = np.random.default_rng(42)
N = 5000

t, p = bgs.mathematics.uniform_sample_sphere(N * 2, rng=rng)
spin_mag = scipy.stats.beta.rvs(
    *bgs.literature.zlochower_dry_spins.values(),
    random_state=rng,
    size=N * 2,
)
spins = bgs.mathematics.convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
s1 = spins[:N, :]
s2 = spins[N:, :]

v = np.full(N, np.nan)

for i, (ss1, ss2) in enumerate(zip(s1, s2)):
    print(f"Sampling {(i+1)/N*100:.1f}% complete...                      ", end="\r")
    # convert unit of spin
    remnant = bgs.literature.ketju_calculate_bh_merger_remnant_properties(
        m1=bh1.m[-1], m2=bh2.m[-1],
        s1=ss1, s2=ss2,
        x1=bh1.x[-1,:], x2=bh2.x[-1,:],
        v1=bh1.v[-1,:], v2=bh2.v[-1,:]
    )
    v[i] = np.linalg.norm(remnant["v"])
print("\nSampling complete")

v = np.sort(v)
P = 1-np.cumsum(v)/np.sum(v)
P = np.clip(P, 1e-7, None)

if True:
    print(f"Quantile corresponding to 1020 km/s is {bgs.mathematics.empirical_cdf(v, 1020)}")
    plt.hist(v, 30, density=True)
    plt.xlabel(r"$v_\mathrm{kick}$")
    plt.show()

if False:
    plt.plot(v, P)
    plt.xlabel(r"$v_\mathrm{kick}$")
    plt.show()

if False:
    rb = lambda vv: 2.02 * (vv/1200)**0.42 + 1
    plt.hist(rb(v), 20, density=True)
    plt.axvline(rb(900), c="tab:red", lw=2)
    plt.ylabel("rb")
    plt.show()