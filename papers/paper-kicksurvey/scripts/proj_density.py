import argparse
from datetime import datetime
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pygad
import baggins as bgs
import dask
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot projected density image for 600km/s case",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-n", "--num", help="snap number", dest="num", default=7, type=int)
parser.add_argument("-z", dest="redshift", help="redshift", type=float, default=0.05)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)

# set the snapshot
snap_file = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")[args.num]
# set the IFU extent
ifu_half_extent = 6

# determine pixel bin width
def get_num_pixels(angscale, spatial_res=0.101, extent=40):
    # spatial res in arcsec/pixel
    return int(extent / (spatial_res * angscale))

snap = pygad.Snapshot(snap_file, physical=True)
# move to CoM frame
pre_ball_mask = pygad.BallMask(5)
centre = pygad.analysis.shrinking_sphere(
    snap.stars,
    pygad.analysis.center_of_mass(snap.stars),
    30,
)
vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
pygad.Translation(-centre).apply(snap, total=True)
pygad.Boost(-vcom).apply(snap, total=True)

density_mask = pygad.ExprMask("abs(pos[:,0]) <= 25") & pygad.ExprMask("abs(pos[:,2]) <= 25")

# plot the density map
fig, ax = plt.subplots(1, 3)#, sharex="all", sharey="all")
fig.set_figwidth(3 * fig.get_figwidth())
fontsize = 12
extent = 40
xaxis = 0
yaxis = 2

def add_extras(ax, pos, ifu_half_extent=ifu_half_extent):
    # make an "aperture" rectangle to show IFU footprint
    ifu_rect = Rectangle(
        (-ifu_half_extent, -ifu_half_extent),
        2 * ifu_half_extent,
        2 * ifu_half_extent,
        fc="none",
        ec="k",
        fill=False,
    )
    ax.add_artist(ifu_rect)
    # mark BH position
    ax.annotate(
        r"$\mathrm{bound\;cluster}$",
        (pos[0,0]+0.25, pos[0,2]-0.25),
        (pos[0,0]+12, pos[0,2]-8),
        color="w",
        arrowprops={"fc":"w", "ec":"w", "arrowstyle":"wedge"},
        ha="right",
        va="bottom",
        fontsize=fontsize
    )
    ax.set_facecolor("k")

ang_scale = bgs.cosmology.angular_scale(args.redshift)
SL.info(f"Angular scale is {ang_scale:.4f} kpc/arcsec")

# figure 1: easy, surface mass density
SL.debug("Plotting surface mass density")
_, _, im, _ = pygad.plotting.image(
    snap.stars[density_mask],
    qty="mass",
    surface_dens=True,
    xaxis=0, yaxis=2,
    cbartitle=r"$\log_{10}\left(\Sigma/\left(\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)\right)$",
    fontsize=fontsize,
    outline=None,
    cmap="rocket",
    ax=ax[0],
    Npx=get_num_pixels(ang_scale),
    extent=extent
)
add_extras(ax=ax[0], pos=snap.bh["pos"])

# figure 2: convert to magnitudes
ax[1].set_facecolor("k")
# let's first define some quantities we will need
filter_code = "Euclid/NISP.Y"
metallicity = 0.012
age = 7.93 # Gyr
bgs.analysis.set_luminosity(snap, metallicity=metallicity, age=age, z=args.redshift)
synth_grid, synth_SED = bgs.analysis.get_spectrum_ssp(age=age, metallicity=metallicity)
euclid_filters = bgs.analysis.get_euclid_filter_collection(synth_grid)

# now we need to bin the galaxy in the 2D plane, 
# and determine the magnitude for each pixel
start_time = datetime.now()
star_count, xedges, yedges = np.histogram2d(
    x = snap.stars[density_mask]["pos"][:, xaxis],
    y = snap.stars[density_mask]["pos"][:, yaxis],
    bins = get_num_pixels(ang_scale),
)
SL.debug(f"There are {(len(xedges)-1)**2:.2e} bins")

# helper function to parallelise magnitude calculation
@dask.delayed
def parallel_mag_helper(num_stars_in_bin):
    return bgs.analysis.get_magnitudes(
        sed=synth_SED,
        stellar_mass=num_stars_in_bin * float(snap.stars["mass"][0]),
        filters_collection=euclid_filters,
        filter_code=filter_code,
        z=args.redshift
    )["app_mag"]

res = []
for sc in star_count.flat:
    res.append(parallel_mag_helper(sc))
res = dask.compute(*res)
mag_map = np.array(res).reshape(star_count.shape).T
SL.info(f"Apparent magnitude calculated in {datetime.now()-start_time}")

im_mag = ax[1].imshow(mag_map, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="mako_r")
im_mag_array = im_mag.get_array()
im_mag_array = im_mag_array[np.isfinite(im_mag_array)]
pygad.plotting.add_cbar(ax=ax[1], cbartitle=r"$\mathrm{App. Mag.}$", clim=np.percentile(im_mag_array, [0.1, 99.9]), cmap=im_mag.get_cmap(), fontcolor="w", fontsize=fontsize)
pygad.plotting.make_scale_indicators(ax=ax[1], extent=pygad.UnitArr(im_mag.get_extent(), units=snap["pos"].units, subs=snap).reshape((2,2)), fontcolor="w", fontsize=fontsize)

# figure 3: plot the S/N map
SL.debug("Plotting S/N map")
ax[2].set_facecolor("k")

filter_window = 10
S_N = np.abs(im_mag.get_array()) / np.abs(median_filter(im_mag.get_array(), size=filter_window, mode="nearest"))
imSN = ax[2].imshow(S_N, origin="lower", extent=im_mag.get_extent(), cmap="mako")
pygad.plotting.make_scale_indicators(ax=ax[2], extent=pygad.UnitArr(im_mag.get_extent(), units=snap["pos"].units, subs=snap).reshape((2,2)), fontcolor="k", fontsize=fontsize)
pygad.plotting.add_cbar(ax[2], cbartitle=r"$S/N$", clim=imSN.get_clim(), cmap=imSN.get_cmap(), fontcolor="k", fontsize=fontsize)
# add a S/N contour
#x_contour = bgs.mathematics.get_histogram_bin_centres(xedges)
#y_contour = bgs.mathematics.get_histogram_bin_centres(yedges)
#CS = ax[2].contour(x_contour, y_contour, S_N, levels=[1], colors="r")
#ax[2].clabel(CS, fontsize=fontsize)

SL.info(f"S/N at BH position is {bgs.mathematics.get_pixel_value_in_image(snap.bh['pos'][0,0], snap.bh['pos'][0,2], imSN)[0]:.2e}")
signal_prom = bgs.analysis.signal_prominence(snap.bh['pos'][0,0], snap.bh['pos'][0,2], imSN, npix=filter_window)
SL.info(f"This corresponds to approx. the {signal_prom:.2f} quantile of pixels nearby")

bgs.plotting.savefig(figure_config.fig_path("density_map.pdf"), fig=fig, force_ext=True)
plt.close()
