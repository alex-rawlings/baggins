import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import baggins as bgs
from ketjugw.units import km_per_s
from tqdm import tqdm
import arviz as az


ESCAPE_VEL = 1800
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
mass_vary = 0.3 # intrinsic scatter of Haring and Rix 2004

t, p = bgs.mathematics.uniform_sample_sphere(N * 2, rng=rng)
spin_mag = scipy.stats.beta.rvs(
    *bgs.literature.zlochower_dry_spins.values(),
    random_state=rng,
    size=N * 2,
)
spins = bgs.mathematics.convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
s1 = spins[:N, :]
s2 = spins[N:, :]

v_mass_constant = np.full(N, np.nan)
v_mass_vary = np.full(N, np.nan)

for i, (ss1, ss2) in tqdm(enumerate(zip(s1, s2)), total=N, desc="Sampling spins"):
    # just sample different spins
    remnant = bgs.literature.ketju_calculate_bh_merger_remnant_properties(
        m1=bh1.m[-1], m2=bh2.m[-1],
        s1=ss1, s2=ss2,
        x1=bh1.x[-1,:], x2=bh2.x[-1,:],
        v1=bh1.v[-1,:], v2=bh2.v[-1,:]
    )
    v_mass_constant[i] = np.linalg.norm(remnant["v"])
    # sample different spins but also different masses
    remnant = bgs.literature.ketju_calculate_bh_merger_remnant_properties(
        m1=bh1.m[-1] * 10**rng.normal(0, mass_vary), 
        m2=bh2.m[-1] * 10**rng.normal(0, mass_vary),
        s1=ss1, s2=ss2,
        x1=bh1.x[-1,:], x2=bh2.x[-1,:],
        v1=bh1.v[-1,:], v2=bh2.v[-1,:]
    )
    v_mass_vary[i] = np.linalg.norm(remnant["v"])

v_mass_constant = np.sort(v_mass_constant)
v_mass_vary = np.sort(v_mass_vary)
P = 1-np.cumsum(v_mass_constant)/np.sum(v_mass_constant)
P = np.clip(P, 1e-7, None)

if True:
    for v, lab in zip((v_mass_constant, v_mass_vary), ("const", "vary")):
        print(f"Quantile corresponding to 1020 km/s is {bgs.mathematics.empirical_cdf(v, 1020)}")
        print(f"Quantile corresponding to 1800 km/s is {bgs.mathematics.empirical_cdf(v, 1800)}")
        plt.hist(v, 30, density=True, label=lab, alpha=0.7)
    plt.xlabel(r"$v_\mathrm{kick}$")
    plt.legend()
    plt.savefig("kick_dist_2.png")

if False:
    plt.plot(v_mass_constant, P)
    plt.xlabel(r"$v_\mathrm{kick}$")
    plt.savefig("kick_dist.png")

if False:
    rb = lambda vv: 2.02 * (vv/1200)**0.42 + 1
    plt.hist(rb(v), 20, density=True)
    plt.axvline(rb(900), c="tab:red", lw=2)
    plt.ylabel("rb")
    plt.show()

if False:
    cols = bgs.plotting.mplColours()
    # test the transform sampling
    fig, ax = plt.subplots(1,2, sharex="all")
    for maxv, axi in zip((1e14, 1020), ax):
        vs = v_mass_constant[v_mass_constant < maxv] / ESCAPE_VEL
        az.plot_dist(2.9 * (vs) ** 0.782 + 1, kind="kde", ax=axi, plot_kwargs={"c":cols[0]})
        '''axi.hist(
            2.9 * (vs) ** 0.782 + 1, 50,
            label=r"$\mathrm{Exponential}$", alpha=0.4
        )'''
        az.plot_dist(3.26 * vs + 1.1, kind="kde", ax=axi, plot_kwargs={"c":cols[1]})
        #az.plot_dist(scipy.stats.norm.rvs(loc=3.26, scale=0.2, random_state=rng, size=len(vs)) * vs + scipy.stats.norm.rvs(loc=1.1, scale=0.07, random_state=rng, size=len(vs)), kind="kde", ax=axi, plot_kwargs={"c":cols[1]})
        #axi.hist(3.26 * vs + 1.1, 50, label=r"$\mathrm{Linear}$", alpha=0.4)
        az.plot_dist(2.47 * (1 - np.exp(-2.62 * vs)) + 0.873, kind="kde", ax=axi, plot_kwargs={"c":cols[2]})
        '''axi.hist(
            2.47 * (1 - np.exp(-2.62 * vs)) + 0.873, 50,
            label=r"$\mathrm{Sigmoid}$", alpha=0.4
        )'''
    plt.show()