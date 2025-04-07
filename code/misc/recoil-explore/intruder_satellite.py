import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs
import merger_ic_generator as mig


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot projected density image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-n", "--new", dest="new", action="store_true", help="create a new IC file"
)
parser.add_argument(
    "-z", "--redshift", dest="redshift", type=float, help="redshift", default=0.6
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

intruder_file = ("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/intruder_ic.hdf5")

if args.new or not os.path.exists(intruder_file):
    SL.warning("Creating a new intruder system!")
    # load and centre snapshot
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_007.hdf5"
    snap = pygad.Snapshot(snapfile, physical=True)
    SL.info(f"Snapshot time is {bgs.general.convert_gadget_time(snap)} Gyr")
    bgs.analysis.basic_snapshot_centring(snap)
    SL.warning(f"BH position is {pygad.utils.geo.dist(snap.bh['pos'])}")

    # get the density of the entire galaxy
    r_edges_gal = np.geomspace(1e-2, 2 * snap.bh["r"][0], 51)
    gal_dens = pygad.analysis.profile_dens(snap.stars, "mass", r_edges=r_edges_gal)

    # extract those stellar particles within the influence radius
    rinfl = list(bgs.analysis.influence_radius(snap).values())[0]
    SL.debug(f"Influence radius is {rinfl:.2e} kpc")
    rinfl_mask = pygad.BallMask(rinfl, snap.bh["pos"][0, :])
    infl_stars = snap.stars[rinfl_mask]
    SL.debug(f"There are {len(infl_stars):.2e} stars within the influence radius")

    # get the density of the total cluster
    def get_cluster_density(Rmax):
        r_edges = np.geomspace(1e-2, Rmax, 21)
        r_centres = bgs.mathematics.get_histogram_bin_centres(r_edges)
        dens = pygad.analysis.profile_dens(
            infl_stars, "mass", r_edges=r_edges, center=snap.bh["pos"].flatten()
        )
        return r_centres, dens

    r_centres_cluster, density_cluster = get_cluster_density(rinfl)
    # only include those stars which are within a radius that encloses a
    # density above the background density
    background_dens = np.interp(
        r_centres_cluster,
        bgs.mathematics.get_histogram_bin_centres(r_edges_gal),
        gal_dens,
    )
    cluster_max_r = r_centres_cluster[
        np.argmin(density_cluster > background_dens) + 1
    ]  # plus 1 so we get the full bin

    # now fit the cluster that is visible above the background
    r_centres_cluster, density_cluster = get_cluster_density(cluster_max_r)
    cluster_mask = pygad.BallMask(cluster_max_r, center=snap.bh["pos"].flatten())
    SL.info(
        f"Updated half mass radius: {bgs.analysis.lagrangian_radius(snap[cluster_mask], 0.5)}"
    )
    SL.info(
        f"Updated LOS velocity dispersion: {pygad.analysis.los_velocity_dispersion(snap.stars[cluster_mask], proj=1)}"
    )

    Mcluster = np.sum(snap.stars[cluster_mask]["mass"])
    SL.debug(f"Cluster mass is {Mcluster:.3e}")
    dehnen_params = bgs.literature.fit_Dehnen_profile(
        r_centres_cluster, density_cluster, Mcluster, bounds=[[0.1, 0.1], [10, 3]]
    )
    SL.info(f"Best fit parameters are: {dehnen_params}")
    quit()
    if args.verbosity == "DEBUG":
        plt.loglog(r_centres_cluster, density_cluster)
        plt.loglog(
            r_centres_cluster,
            bgs.literature.Dehnen(r_centres_cluster, *dehnen_params, Mcluster),
            label=f"a: {dehnen_params[0]:.3f}, g: {dehnen_params[1]:.3f}",
        )
        plt.legend()
        bgs.plotting.savefig(
            os.path.join(bgs.FIGDIR, "kick-survey/dehnen.png"), force_ext=True
        )
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
            mig.Translation([0, 0, -15]),
        ),
    )
    mig.write_hdf5_ic_file(intruder_file, joined_system, double_precision=True)

# read in intruder snapshot -> should already be centred
snap = pygad.Snapshot(intruder_file, physical=True)
fig, ax = plt.subplots(1, 3)

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
    extent=muse_nfm.max_extent,
)
ax[0].annotate(
    r"$\mathrm{with\;SMBH}$",
    (snap.bh["pos"][0, 0] + 0.25, snap.bh["pos"][0, 2] - 0.25),
    (snap.bh["pos"][0, 0] + 11, snap.bh["pos"][0, 2] - 8),
    color="w",
    arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
    ha="right",
    va="bottom",
    fontsize=fontsize,
)
ax[0].annotate(
    r"$\mathrm{without\;SMBH}$",
    (-0.5, -15),
    (-10, 2),
    color="w",
    arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
    ha="center",
    va="bottom",
    fontsize=fontsize,
)
"""# make an "aperture" rectangle to show IFU footprint
ifu_rect = Rectangle(
    (-muse_nfm.extent * 0.5, -muse_nfm.extent * 0.5),
    muse_nfm.extent,
    muse_nfm.extent,
    fc="none",
    ec="k",
    fill=False,
)
ax[0].add_artist(ifu_rect)
ax[0].set_facecolor("k")"""

# create IFU maps
seeing = seeing = {
    "num": 25,
    "sigma": muse_nfm.pixel_width,
    "rng": np.random.default_rng(42),
}
voronoi = bgs.analysis.VoronoiKinematics(
    x=snap.stars[ifu_mask]["pos"][:, 0],
    y=snap.stars[ifu_mask]["pos"][:, 2],
    V=snap.stars[ifu_mask]["vel"][:, 1],
    m=snap.stars[ifu_mask]["mass"],
    Npx=muse_nfm.number_pixels,
    seeing=seeing,
)
voronoi.make_grid(part_per_bin=seeing["num"] * 5000)
voronoi.binned_LOSV_statistics()
voronoi.plot_kinematic_maps(ax=ax[1:], cbar="inset", fontsize=fontsize)
for axi in ax[1:]:
    axi.set_xticks([])
    axi.set_xticklabels([])
    axi.set_yticks([])
    axi.set_yticklabels([])
    bgs.plotting.draw_sizebar(
        axi,
        10,
        "kpc",
        location="lower left",
        color="k",
        size_vertical=0.4,
        textsize=fontsize,
    )
    axi.scatter(
        snap.bh["pos"][:, 0],
        snap.bh["pos"][:, 2],
        lw=1,
        s=150,
        ec="k",
        fc="none",
    )

#bgs.plotting.savefig(figure_config.fig_path("intruder.pdf"), force_ext=True)
