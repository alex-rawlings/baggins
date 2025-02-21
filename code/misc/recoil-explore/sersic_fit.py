import numpy as np
import scipy.optimize as so
import scipy.stats as ss
import matplotlib.pyplot as plt
import baggins as bgs

bgs.plotting.check_backend()

data = bgs.utils.load_data("mag_data.pickle")
mags, bh, xedges, yedges = data["mags"], data["bh"], data["xedges"], data["yedges"]

bh_rad = np.sqrt(bh[0]**2 + bh[1]**2)
bh_pix = [np.digitize(bh[0], xedges), np.digitize(bh[1], yedges)]

print(f"BH Pixel: {bh_pix}")
bh_pix_lin = bh_pix[0] * len(xedges) + bh_pix[1]

flux = np.log(bgs.analysis.get_flux_from_magnitude(mags.flatten()))

x, y = bgs.mathematics.get_histogram_bin_centres(xedges), bgs.mathematics.get_histogram_bin_centres(yedges)

XX, YY = np.meshgrid(x, y)
R = np.sqrt(XX**2 + YY**2).flatten()

def log_sersic(R, logI0, Re, n):
    return logI0 - bgs.general.sersic_b_param(n) * ((R/Re)**(1/n) - 1)


# fit the params
params, pcov, *_ = so.curve_fit(log_sersic, R, flux, bounds=([-10, 1, 1], [10, 20, 20]), p0=[0, 7, 4])

print(params)
print(pcov)

fig, ax = plt.subplots(2, 1, sharex="all")
ax[0].set_xlim(1e-1, 40)

rs = np.geomspace(5e-2, 2*R.max(), 400)
ax[0].loglog(rs, np.exp(log_sersic(rs, *params)))
ax[0].scatter(R, np.exp(flux), marker=".", c="tab:red")

res = flux-log_sersic(R, *params)
tdist = ss.t(len(flux)-3)

Tvals = tdist.ppf(res)

ax[1].scatter(R, res, marker=".")
ax[1].scatter(R[bh_pix_lin], res[bh_pix_lin], marker="x", c="tab:red")
ax[1].axvline(bh_rad, c="k", alpha=0.3, zorder=0.1)
plt.savefig("sersic.png", dpi=300)