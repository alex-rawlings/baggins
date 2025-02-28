import numpy as np
import scipy.optimize as so
import scipy.stats as ss
import matplotlib.pyplot as plt
import baggins as bgs

bgs.plotting.check_backend()

data = bgs.utils.load_data("mag_data.pickle")
mags, bh, xedges, yedges = data["mags"], data["bh"], data["xedges"], data["yedges"]

pixel_widths = np.diff(xedges)[0], np.diff(yedges)[0]
print(pixel_widths)
bh_pix = [np.digitize(bh[0], xedges), np.digitize(bh[1], yedges)]

#plt.imshow(mags, extent=[min(xedges), max(xedges), min(yedges), max(yedges)])
#plt.show()

print(f"BH Pixel: {bh_pix}")
# minus 1's as R is constructed on the centres of the bins, not the bin edges
bh_pix_lin = (bh_pix[1]-1) * (len(xedges)-1) + (bh_pix[0]-1)
bh_rad = np.sqrt(bh[0]**2 + bh[1]**2)
print(f"BH radius: {bh_rad}")

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
ax[0].plot(R, np.exp(flux), marker=".", ls="")
ax[0].loglog(rs, np.exp(log_sersic(rs, *params)))

res = flux-log_sersic(R, *params)
tdist = ss.t(len(flux)-3)

Tvals = tdist.ppf(res)

ax[1].scatter(R, res, marker=".")
ax[1].scatter(R[bh_pix_lin], res[bh_pix_lin], marker="x", label="BH pixel")
ax[1].axvline(bh_rad, c="k", alpha=0.3, zorder=0.1, label="True BH radius")
ax[1].axhline(0, ls=":", c="k", zorder=0.1)

ax[0].set_ylabel("Flux")
ax[1].set_ylabel("Residuals (normalised to t-Distribution)")
ax[1].set_xlabel("radii")
ax[1].legend()
plt.savefig("sersic.png", dpi=300)
