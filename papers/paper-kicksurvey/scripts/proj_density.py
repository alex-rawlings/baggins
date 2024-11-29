import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pygad
import baggins as bgs
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot projected density image for 600km/s case",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-n", "--num", help="snap number", dest="num", default=7, type=int)
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
euclid_resolution = 0.101 # arcsec/pixel

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
fig, ax = plt.subplots(1, 3, sharex="all", sharey="all")
fig.set_figwidth(3 * fig.get_figwidth())
fontsize = 12
extent = 40

for axi, z, angscale in zip(ax, (0.02, 0.1, 1), (0.405285, 1.8443, 8.00871)):
    # Npx = X kpc/(arcsec/pix * kpc/arcsec)
    Npx = int(extent / (euclid_resolution * angscale))
    pygad.plotting.image(
        snap.stars[density_mask],
        qty="mass",
        surface_dens=True,
        xaxis=0, yaxis=2,
        cbartitle=r"$\log_{10}\left(\Sigma/\left(\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)\right)$",
        fontsize=fontsize,
        outline=None,
        cmap="rocket",
        ax=axi,
        Npx=Npx,
        extent=extent
        #vlim = (10**6.4, 10**9.4)
    )
    axi.text(10, -17.5, f"$z={z:.2f}$", color="w", fontsize=fontsize)
    axi.set_facecolor("k")

    # make an "aperture" rectangle to show IFU footprint
    ifu_rect = Rectangle(
        (-ifu_half_extent, -ifu_half_extent),
        2 * ifu_half_extent,
        2 * ifu_half_extent,
        fc="none",
        ec="k",
        fill=False,
    )
    axi.add_artist(ifu_rect)

    # mark BH position
    axi.annotate(
        r"$\mathrm{bound\;cluster}$",
        (snap.bh["pos"][0,0]+0.25, snap.bh["pos"][0,2]-0.25),
        (snap.bh["pos"][0,0]+12, snap.bh["pos"][0,2]-8),
        color="w",
        arrowprops={"fc":"w", "ec":"w", "arrowstyle":"wedge"},
        ha="right",
        va="bottom",
        fontsize=fontsize
    )
    SL.debug(f"Map plotted for z={z:.2f}")

bgs.plotting.savefig(figure_config.fig_path("density_map.pdf"), force_ext=True)
