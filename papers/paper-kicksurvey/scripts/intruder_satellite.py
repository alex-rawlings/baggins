import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
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
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_005.hdf5"
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
fig, ax = plt.subplots(1, 3)
fig.set_figwidth(3 * fig.get_figwidth())

# plot the surface mass density
pygad.plotting.image(
    snap.stars[pygad.BallMask(30)],
    qty="mass",
    surface_dens=True,
    ax=ax[0],
    cmap="rocket",
    cbartitle=r"$\log_{10}\left(\Sigma/\left(\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)\right)$",
    outline=None,
)
ax[0].annotate(
    r"$\mathrm{with\;SMBH}$",
    (snap.bh["pos"][0, 0] + 0.25, snap.bh["pos"][0, 2] - 0.25),
    (snap.bh["pos"][0, 0] + 12, snap.bh["pos"][0, 2] - 8),
    color="w",
    arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
    ha="right",
    va="bottom",
    fontsize=12,
)
ax[0].annotate(
    r"$\mathrm{without\;SMBH}$",
    (-snap.bh["pos"][0, 0] + 0.25, snap.bh["pos"][0, 2] - 0.5),
    (-snap.bh["pos"][0, 0] - 1, snap.bh["pos"][0, 2] - 8),
    color="w",
    arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
    ha="center",
    va="bottom",
    fontsize=12,
)
ax[0].set_facecolor("k")

bgs.plotting.savefig(figure_config.fig_path("intruder.pdf"), force_ext=True)
