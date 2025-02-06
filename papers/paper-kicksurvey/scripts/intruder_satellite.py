import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pygad
import baggins as bgs
import merger_ic_generator as mig
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot projected density image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-n", "--new", dest="new", action="store_true", help="create a new IC file"
)
parser.add_argument(
    "-z", "--redshift", dest="redshift", type=float, help="redshift", default=0.3
)
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

intruder_file = figure_config.data_path("intruder_ic.hdf5")

if args.new or not os.path.exists(intruder_file):
    SL.warning("Creating a new intruder system!")
    # load and centre snapshot
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_006.hdf5"
    snap = pygad.Snapshot(snapfile, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)
    SL.warning(f"BH position is {pygad.utils.geo.dist(snap.bh['pos'])}")

    # extract those stellar particles within the influence radius
    rinfl = list(bgs.analysis.influence_radius(snap).values())[0]
    SL.debug(f"Influence radius is {rinfl:.2e} kpc")
    rinfl_mask = pygad.BallMask(rinfl, snap.bh["pos"][0, :])
    infl_stars = snap.stars[rinfl_mask]
    SL.debug(f"There are {len(infl_stars):.2e} stars within the influence radius")

    # fit a Dehnen density profile to these stars
    redges = np.geomspace(1e-2, 2, 15)
    rcentres = bgs.mathematics.get_histogram_bin_centres(redges)
    density = pygad.analysis.profile_dens(
        infl_stars, "mass", r_edges=redges, center=snap.bh["pos"].flatten()
    )
    Mcluster = np.sum(infl_stars["mass"])
    dehnen_params = bgs.literature.fit_Dehnen_profile(
        rcentres, density, Mcluster, bounds=[[0.1, 0.1], [10, 3]]
    )
    SL.info(f"Best fit parameters are: {dehnen_params}")

    if args.verbosity == "DEBUG":
        plt.loglog(rcentres, density)
        plt.loglog(rcentres, bgs.literature.Dehnen(rcentres, *dehnen_params, Mcluster))
        bgs.plotting.savefig("dehnen.png", force_ext=True)
        plt.close()

    # create a new cluster with the same density profile
    gadget_mass_scaling = 1e10
    new_cluster = mig.ErgodicSphericalSystem(
        mig.DehnenSphere(
            mass=float(Mcluster) / gadget_mass_scaling,
            scale_radius=dehnen_params[0],
            gamma=dehnen_params[1],
            particle_mass=float(snap.stars["mass"][0]) / gadget_mass_scaling,
            particle_type=mig.ParticleType.STARS,
        )
    )
    merger_remnant = mig.SnapshotSystem(snapfile, center_CoM=False)
    joined_system = mig.JoinedSystem(
        merger_remnant,
        mig.TransformedSystem(
            new_cluster,
            mig.Translation(snap.bh["pos"].flatten() * np.array([-1, 1, 1])),
        ),
    )
    mig.write_hdf5_ic_file(intruder_file, joined_system, double_precision=True)

# read in intruder snapshot -> should already be centred
snap = pygad.Snapshot(intruder_file, physical=True)
fig, ax = plt.subplots(
    1, 3, gridspec_kw={"width_ratios": [1, 1.1, 1.1], "wspace": 0.02}
)
fig.set_figwidth(3 * fig.get_figwidth())

# some quantities common to the subplots
muse_nfm = bgs.analysis.MUSE_NFM()
muse_nfm.redshift = args.redshift
SL.info(f"Extent is {muse_nfm.extent:.2e} kpc")
SL.info(f"Pixel width: {muse_nfm.pixel_width:.2e} kpc")
ifu_mask = pygad.ExprMask(f"abs(pos[:,0]) <= {muse_nfm.extent*0.5}") & pygad.ExprMask(
    f"abs(pos[:,2]) <= {muse_nfm.extent*0.5}"
)
fontsize = 12

# plot the surface mass density
pygad.plotting.image(
    snap.stars[pygad.BallMask(muse_nfm.max_extent)],
    qty="mass",
    surface_dens=True,
    xaxis=0,
    yaxis=2,
    ax=ax[0],
    cmap="rocket",
    cbartitle=r"$\log_{10}\left(\Sigma/\left(\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)\right)$",
    outline=None,
    fontsize=fontsize,
)
ax[0].annotate(
    r"$\mathrm{with\;SMBH}$",
    (snap.bh["pos"][0, 0] + 0.25, snap.bh["pos"][0, 2] - 0.25),
    (snap.bh["pos"][0, 0] + 12, snap.bh["pos"][0, 2] - 8),
    color="w",
    arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
    ha="right",
    va="bottom",
    fontsize=fontsize,
)
ax[0].annotate(
    r"$\mathrm{without\;SMBH}$",
    (-snap.bh["pos"][0, 0] + 0.25, snap.bh["pos"][0, 2] - 0.5),
    (-snap.bh["pos"][0, 0] - 1, snap.bh["pos"][0, 2] - 8),
    color="w",
    arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
    ha="center",
    va="bottom",
    fontsize=fontsize,
)
# make an "aperture" rectangle to show IFU footprint
ifu_rect = Rectangle(
    (-muse_nfm.extent * 0.5, -muse_nfm.extent * 0.5),
    muse_nfm.extent,
    muse_nfm.extent,
    fc="none",
    ec="k",
    fill=False,
)
ax[0].add_artist(ifu_rect)
ax[0].set_facecolor("k")

# create IFU maps
seeing = seeing = {
    "num": 25,
    "sigma": muse_nfm.pixel_width,
    "rng": np.random.default_rng(42),
}
voronoi_stats = bgs.analysis.voronoi_binned_los_V_statistics(
    x=snap.stars[ifu_mask]["pos"][:, 0],
    y=snap.stars[ifu_mask]["pos"][:, 2],
    V=snap.stars[ifu_mask]["vel"][:, 1],
    m=snap.stars[ifu_mask]["mass"],
    Npx=muse_nfm.number_pixels,
    part_per_bin=seeing["num"] * 5000,
    seeing=seeing,
)
bgs.plotting.voronoi_plot(voronoi_stats, ax=ax[1:])
for axi in ax[1:]:
    axi.set_xticks([])
    axi.set_xticklabels([])
    axi.set_yticks([])
    axi.set_yticklabels([])
    bgs.plotting.draw_sizebar(
        axi,
        5,
        "kpc",
        location="lower left",
        color="k",
        size_vertical=0.1,
        textsize=fontsize,
    )

bgs.plotting.savefig(figure_config.fig_path("intruder.pdf"), force_ext=True)
