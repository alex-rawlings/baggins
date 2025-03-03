import numpy as np
import scipy.optimize as so
import scipy.stats as ss
from scipy.ndimage import uniform_filter, generic_filter
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import seaborn as sns
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
print(f"BH radial pos: {bh_rad}")

flux = np.log(bgs.analysis.get_flux_from_magnitude(mags.flatten()))

x, y = bgs.mathematics.get_histogram_bin_centres(xedges), bgs.mathematics.get_histogram_bin_centres(yedges)

XX, YY = np.meshgrid(x, y)
R = np.sqrt(XX**2 + YY**2).flatten()

def log_sersic(R, logI0, Re, n):
    return logI0 - bgs.general.sersic_b_param(n) * ((R/Re)**(1/n) - 1)

def ellipse(x, y, x0, y0, theta, dt, ellip, boxy):
    _Rt = np.sqrt((x-x0)**2 + (y-y0)**2)
    _thetat = np.arctan2(x-x0, y-y0) + theta
    _thetat = _thetat + np.sin(dt*_Rt/np.max(_Rt)*np.pi)
    Arat = 1 - ellip
    Rm = (np.abs(_Rt * np.sin(_thetat) * Arat) ** (2 + boxy) +
            np.abs(_Rt * np.cos(_thetat)) ** (2 + boxy)
        )**(1 / (2 + boxy))
    return Rm

def log_sersic_2d(xy, x0, y0, theta, dt, ellip, boxy, logI0, Re, n):
    x , y = xy[0].flatten(), xy[1].flatten()
    Rm = ellipse(x, y, x0, y0, theta, dt, ellip, boxy)
    return log_sersic(Rm, logI0, Re, n)



# fit the params
params, pcov, *_ = so.curve_fit(log_sersic_2d,
                                (XX, YY),
                                flux,
                                bounds=(
                                    [-5, -5, 0, -1, 0, 0, -10, 2, 1],
                                    [5, 5, 2*np.pi, 1, 1, 1, 10, 20, 20]
                                ),
                                p0=[1e-10, 1e-10, 1e-10, 1e-2, 1e-2, 1e-2, 1, 7, 4]
                                )

print(params)

fig, ax = plt.subplots(1, 3, sharex="all", sharey="all")
fig.set_figwidth(2*fig.get_figwidth())


ax[0].imshow(mags, origin="lower", extent=[min(x), max(x), min(y), max(y)], cmap="cividis", aspect="equal")
ax[0].contour(XX, YY, mags, levels=10, colors="k", linestyles="solid", linewdiths=0.5)

xx = np.linspace(np.min(x), np.max(x), 400)
yy = np.linspace(np.min(y), np.max(y), 400)
#ax[1].plot(xx, yy, np.exp(flux), marker=".", ls="")
#ax[1].loglog(rs, np.exp(log_sersic(rs, *params)))
imfit = log_sersic_2d((XX, YY), *params).reshape(XX.shape)
ax[1].imshow(imfit, origin="lower", extent=[min(x), max(x), min(y), max(y)], cmap="cividis", aspect="equal")
for axi in ax[1:]:
    axi.contour(XX, YY, imfit, levels=10, colors="k", linestyles="solid", linewdiths=0.5)

# get the residuals
res = flux-log_sersic_2d((XX, YY), *params)
res = res.reshape(XX.shape)

# find signal in the residuals
# determine the prominence within some aperture
filter_kwargs = {"size": 5, "mode": "nearest"}
prom = -(
    res - uniform_filter(res, **filter_kwargs)
) / generic_filter(
    res,
    lambda x: np.nan if np.any(np.isnan(x)) else np.std(x),
    **filter_kwargs,
)
prom[prom < 0] = 0

res_standard = (res - np.nanmean(res)) / np.nanstd(res)

print(f"Residual at BH pos: {res_standard[bh_pix[0], bh_pix[1]]}")

cmap = sns.cubehelix_palette(
    start=0.7,
    rot=-0.5,
    gamma=0.3,
    hue=1.0,
    light=1,
    dark=0,
    reverse=False,
    as_cmap=True,
)
im_SN = ax[2].imshow(
    res_standard,
    origin="lower",
    extent=[min(x), max(x), min(y), max(y)],
    cmap="vlag",
    aspect="equal",
    norm=CenteredNorm(vcenter=0)
)
plt.colorbar(im_SN, label="prom", ax=ax[2])

phi = np.linspace(0, 2*np.pi, 400)
ax[2].scatter(*bh, marker="o", s=100, ec="k", fc="none", lw=0.5)

'''tdist = ss.t(len(flux)-3)

Tvals = tdist.ppf(res)

ax[2].scatter(R, res, marker=".")
ax[2].scatter(R[bh_pix_lin], res[bh_pix_lin], marker="x", label="BH pixel")
ax[2].axvline(bh_rad, c="k", alpha=0.3, zorder=0.1, label="True BH radius")
ax[2].axhline(0, ls=":", c="k", zorder=0.1)

ax[1].set_ylabel("Flux")
ax[2].set_ylabel("Residuals (normalised to t-Distribution)")
ax[2].set_xlabel("radii")
ax[2].legend()'''
plt.savefig("sersic.png", dpi=300)
