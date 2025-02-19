import numpy as np
import scipy.stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import baggins as bgs
from ketjugw.units import km_per_s

rng = np.random.default_rng(8767857)
premerger_ketjufile = bgs.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output")[0]

bh1, bh2, *_ = bgs.analysis.get_bound_binary(premerger_ketjufile)
# move to Gadget units: kpc, km/s, 1e10Msol
bh1.x /= bgs.general.units.kpc
bh2.x /= bgs.general.units.kpc
bh1.v /= km_per_s
bh2.v /= km_per_s
bh1.m /= 1e10
bh2.m /= 1e10
bh1 = bh1[-1]
bh2 = bh2[-1]

def _spin_setter_random(nn):
    t, p = bgs.mathematics.uniform_sample_sphere(nn * 2, rng=rng)
    # XXX: need to sample the full grid, not just the diagonal
    #t = np.arccos(2 * np.linspace(0, 1, nn) - 1)
    #p = 2 * np.pi * np.linspace(0, 1, nn)
    spin_mag = scipy.stats.beta.rvs(
        *bgs.literature.zlochower_dry_spins.values(),
        random_state=rng,
        size=nn * 2,
    )
    #spin_mag=[0.99] * nn * 2
    spins = bgs.mathematics.convert_spherical_to_cartesian(np.vstack((spin_mag, t, p)).T)
    return spins[:nn, :], spins[nn:, :]


def _spin_setter_det(nn, t2):
    t = np.zeros(2*nn)
    t[nn:] = np.full(nn, t2)
    spin_mag = scipy.stats.beta.rvs(
        *bgs.literature.zlochower_dry_spins.values(),
        random_state=rng,
        size=nn * 2,
    )
    spins = bgs.mathematics.convert_spherical_to_cartesian(np.vstack((spin_mag, t, np.zeros(2*nn))).T)
    return spins[:nn, :], spins[nn:, :]

def sample_spins(N=1e4, theta2=None):
    N = int(N)
    vkick = np.full(N, np.nan)
    angle_between_vectors = np.full_like(vkick, np.nan)
    mass_ratio = np.full_like(vkick, np.nan)
    spin1 = np.full_like(vkick, np.nan)
    spin2 = np.full_like(vkick, np.nan)
    remaining = np.ones(N, dtype=bool)
    iters = 0
    max_iters = 100
    while np.any(remaining) and iters < max_iters:
        # generate spins
        if theta2 is None:
            s1, s2 = _spin_setter_random(np.sum(remaining))
        else:
            s1, s2 = _spin_setter_det(np.sum(remaining), theta2)
        update_idxs = np.where(remaining == 1)[0]
        for i, (ss1, ss2) in tqdm(
            enumerate(zip(s1, s2)),
            total=len(s1),
            desc=f"Sampling BH spins (iteration {iters})",
        ):
            m1 = bh1.m * 10**rng.normal(0, 0.3)
            m2 = bh2.m * 10**rng.normal(0, 0.3)
            remnant = bgs.literature.ketju_calculate_bh_merger_remnant_properties(
                m1=m1,
                m2=m2,
                s1=ss1,
                s2=ss2,
                x1=bh1.x.flatten(),
                x2=bh2.x.flatten(),
                v1=bh1.v.flatten(),
                v2=bh2.v.flatten(),
            )
            vkick[update_idxs[i]] = np.linalg.norm(remnant["v"])
            angle_between_vectors[update_idxs[i]] = np.cos(bgs.mathematics.angle_between_vectors(ss1, ss2))
            mass_ratio[update_idxs[i]] = m1/m2 if m1 < m2 else m2/m1
            spin1[update_idxs[i]] = bgs.mathematics.radial_separation(ss1)
            spin2[update_idxs[i]] = bgs.mathematics.radial_separation(ss2)
        remaining = np.isnan(vkick)
        iters += 1
    vkick = np.log10(vkick)
    mass_ratio = np.log10(mass_ratio)
    assert np.all(np.abs(angle_between_vectors) <= 1.)
    assert ~np.any(np.isnan(vkick)) and ~np.any(np.isnan(angle_between_vectors)) and ~np.any(np.isnan(mass_ratio))
    return vkick, angle_between_vectors, mass_ratio, spin1, spin2

fig, ax = plt.subplots(1,4, sharex="all", sharey="all")

for axi, theta2 in zip(ax.flat, np.linspace(0, np.pi/2, len(ax))):
    vkick, angle_between_vectors, mass_ratio, spin1, spin2 = sample_spins(1e4, theta2=theta2)
    spin_bins = np.linspace(0, 1, 21)
    binned_vkick_med, xedges, yedges, binnum = scipy.stats.binned_statistic_2d(
        spin1,
        spin2,
        vkick,
        statistic="median",
        bins=(spin_bins, spin_bins)
    )

    XX, YY = np.meshgrid(
        bgs.mathematics.get_histogram_bin_centres(spin_bins), 
        bgs.mathematics.get_histogram_bin_centres(spin_bins)
    )

    p = axi.imshow(binned_vkick_med, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), aspect="auto", vmin=0, vmax=3.5)
    axi.set_xlabel(r"$\chi_1$")
    axi.set_title(r"$\theta_2=$"+f"{theta2*180/np.pi:.1f} degrees")
    CS = axi.contour(XX, YY, binned_vkick_med, levels=[np.log10(270)])
    axi.clabel(CS, inline=True)
ax[0].set_ylabel(r"$\chi_2$")
plt.colorbar(p, ax=ax[-1], label="med(log(vkick))")
plt.show()

quit()
print(f"cos(theta) varies from {np.min(angle_between_vectors):.2f} to {np.max(angle_between_vectors):.2f}")

spin_bins = np.linspace(-1, 1, 21)
q_bins = np.linspace(np.min(mass_ratio), np.max(mass_ratio), 21)


binned_vkick_count, xedges, yedges, binnum = scipy.stats.binned_statistic_2d(angle_between_vectors, mass_ratio, vkick, bins=(spin_bins, q_bins))

binned_vkick_med, xedges, yedges, binnum = scipy.stats.binned_statistic_2d(angle_between_vectors, mass_ratio, vkick, statistic="median", bins=(spin_bins, q_bins))

print(f"x bin edges are {xedges[0]:.2f} to {xedges[-1]:.2f}")

binned_vkick_std, *_ = scipy.stats.binned_statistic_2d(angle_between_vectors, mass_ratio, vkick, statistic="std", bins=(spin_bins, q_bins))

XX, YY = np.meshgrid(
    bgs.mathematics.get_histogram_bin_centres(spin_bins), 
    bgs.mathematics.get_histogram_bin_centres(q_bins)
    )

'''plt.imshow(binned_vkick, origin="lower", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))
CS = plt.contour(XX, YY, binned_vkick, levels=[np.log10(270)])
plt.clabel(CS, inline=True)
plt.show()'''


fig, ax = plt.subplots(3,1,sharex="all", sharey="all")
ax[-1].set_xlabel(r"cos($\theta$)")
for i in range(3):
    ax[i].set_ylabel(r"log(m1/m2)")
# counts
p = ax[0].imshow(binned_vkick_count, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), aspect="auto")
plt.colorbar(p, ax=ax[0], label="count")

# median recoil
p = ax[1].imshow(binned_vkick_count, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), aspect="auto")
CS = ax[1].contour(XX, YY, binned_vkick_med, levels=[np.log10(270)])
ax[1].clabel(CS, inline=True)
plt.colorbar(p, ax=ax[1], label="med(log(vkick))")

# std of recoil
p = ax[2].imshow(binned_vkick_std, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), aspect="auto")
plt.colorbar(p, ax=ax[2], label="std(log(vkick))")

bgs.plotting.savefig("vkick_dependence.png")
plt.show()
