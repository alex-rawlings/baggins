from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import baggins as bgs

bgs.plotting.check_backend()

# XXX: This script is taken directly from:
# bgs.analysis.GaussianProcesses.VkickApocentreGP._set_stan_data_OOS()

kdirs = dict(
    minor="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/minor_merger/H_2M_b-H_2M_minor_d-30.000-1.900",
    major="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000",
)

rng = np.random.default_rng(42)
fig, ax = plt.subplots()
ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$")
ax.set_ylabel(r"$\mathrm{CDF}$")

bins = np.arange(0, 4001, 100)

for k, v in kdirs.items():
    print(f"Doing {k}")
    ketju_file = bgs.utils.get_ketjubhs_in_dir(v)[0]

    # get the last output as a binary
    bh1, bh2, _ = bgs.analysis.get_bound_binary(ketju_file)
    bh1 = bh1[-1]
    bh2 = bh2[-1]

    # randomly sample recoil velocities from Zlochower Lousto
    spins = bgs.literature.SMBHSpins("zlochower_dry", "skewed", rng=rng)
    L = ketjugw.orbital_angular_momentum(bh1, bh2).flatten()

    N = 10000

    s1 = spins.sample_spins(N, L=L)
    s2 = spins.sample_spins(N, L=L)
    vkick = np.full(N, np.nan)

    # transform to Gadget units
    bh1.x /= bgs.general.units.kpc
    bh2.x /= bgs.general.units.kpc
    bh1.v /= ketjugw.units.km_per_s
    bh2.v /= ketjugw.units.km_per_s
    bh1.m /= 1e10
    bh2.m /= 1e10

    for i, (ss1, ss2) in tqdm(
        enumerate(zip(s1, s2)), total=len(s1), desc="Sampling kicks"
    ):
        remnant = bgs.literature.ketju_calculate_bh_merger_remnant_properties(
            m1=bh1.m,
            m2=bh2.m,
            s1=ss1,
            s2=ss2,
            x1=bh1.x.flatten(),
            x2=bh2.x.flatten(),
            v1=bh1.v.flatten(),
            v2=bh2.v.flatten(),
        )
        vkick[i] = np.linalg.norm(remnant["v"])

    # print some quantiles
    qs = [0.5, 0.9, 0.95]
    quants = np.nanquantile(vkick, qs)
    for q, quant in zip(qs, quants):
        print(f"{q:.2f} quantile is {quant:.2e}")

    # ax.hist(vkick, bins=bins, label=f"$\mathrm{{{k}}}$", cumulative=True, density=True)
    ecdf = bgs.mathematics.EmpiricalCDF(vkick)
    ecdf.plot(ax=ax, label=k, ci_prob=0.99)
ax.legend()
# bgs.plotting.savefig(figure_config.fig_path("major_minor.pdf"), force_ext=True)
