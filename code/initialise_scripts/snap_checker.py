import argparse
import os.path
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Check key quantities of snapshot",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(type=str, help="path to snapshot", dest="snap")
parser.add_argument(
    "-s", "--save", type=str, help="figure save location", default=None, dest="savefig"
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

snap = pygad.Snapshot(args.snap, physical=True)
centre = pygad.analysis.shrinking_sphere(snap, pygad.analysis.center_of_mass(snap), 30)
T = pygad.Translation(-centre)
T.apply(snap, total=True)
valid_fams = {pt for pt in ["stars", "bh", "dm", "gas"] if pt in snap.families()}

# print what families there are
SL.info(f"This snapshot contain the particle families {snap.families()}")

# default marker style
mkwargs = {
    "marker": "o",
    "ls": "",
    "mec": "k",
    "mew": 0.5,
    "markersize": 10,
    "zorder": 6,
}

# print some masses
total_mass = 0
for f in valid_fams:
    species = getattr(snap, f)
    if len(np.unique(species["mass"])) == 1:
        SL.info(f"Mass of {f} particle is {species['mass'][0]:.3e}")
    else:
        SL.info(
            f"Mass of {f} particle varies from {min(species['mass']):.3e} to {max(species['mass']):.3e}"
        )
    species_mass = sum(species["mass"])
    SL.info(f"Combined {f} mass: {species_mass:.3e}")
    total_mass += species_mass
SL.info(f"Total mass of galaxy: {total_mass:.3e}")

# density profiles
valid_fams_wo_bh = valid_fams.difference({"bh"})
fig_dens, ax_dens = plt.subplots(
    1, len(valid_fams_wo_bh), figsize=(4 * len(valid_fams_wo_bh), 4), sharex="all"
)
Rs = {"stars": 100, "dm": 1000, "gas": 100}
for i, f in enumerate(valid_fams_wo_bh):
    SL.debug(f"Plotting 3D density for {f}...")
    pygad.plotting.profile(
        getattr(snap, f),
        Rs[f],
        qty="mass",
        N=50,
        logbin=True,
        logscale=True,
        minlog=Rs[f] / 1e4,
        ax=ax_dens[i],
        lw=3,
    )
    ax_dens[i].set_title(f)
if args.savefig is not None:
    bgs.plotting.savefig(os.path.join(args.savefig, "densities.png"), fig=fig_dens)

if "gas" in valid_fams:
    # phase diagram if gas
    SL.debug("Creating phase diagram...")
    fig_phase, ax_phase, *_ = pygad.plotting.phase_diagram(snap.gas, showcbar=True)
    # mark a box for star forming region
    current_xlims = ax_phase.get_xlim()
    current_ylims = ax_phase.get_ylim()
    threshold_kwargs = {"colors": "tab:red", "lw": 2}
    dens_threshold = np.log10(2.2e-24)
    temp_threshold = np.log10(1.2e4)
    if current_ylims[1] > temp_threshold and current_xlims[1] > dens_threshold:
        ax_phase.vlines(
            dens_threshold,
            ymin=current_ylims[0],
            ymax=temp_threshold,
            **threshold_kwargs,
        )
        ax_phase.hlines(
            temp_threshold,
            xmin=dens_threshold,
            xmax=current_xlims[1],
            **threshold_kwargs,
        )
    else:
        SL.info(
            f"Temperature threshold not plotted (threshold: {temp_threshold:.1e}, ymax: {current_ylims[1]:.1e})"
        )
        SL.info(
            f"Density threshold not plotted (threshold: {dens_threshold:.1e}, xmax: {current_xlims[1]:.1e})"
        )
    if args.savefig is not None:
        bgs.plotting.savefig(os.path.join(args.savefig, "phase.png"), fig=fig_phase)

    # metallicity gradient if gas
    SL.debug("Creating metallicity profile...")
    fig_met, ax_met = plt.subplots(1, 1)
    for _fam in ("stars", "gas"):
        subsnap = getattr(snap, _fam)
        stat, edges, *_ = scipy.stats.binned_statistic(
            np.log10(subsnap["r"]),
            subsnap["metallicity"] / pygad.physics.solar.Z(),
            bins=50,
        )
        ax_met.plot(
            bgs.mathematics.get_histogram_bin_centres(edges), stat, lw=2, label=_fam
        )
    ax_met.legend()
    ax_met.set_xlabel(r"$\log(r/kpc)$")
    ax_met.set_ylabel(r"$Z/Z_\odot$")
    if args.savefig:
        bgs.plotting.savefig(os.path.join(args.savefig, "metallicity.png"), fig=fig_met)

# half mass and effective radius
if "stars" in valid_fams:
    r_half = pygad.analysis.half_mass_radius(snap.stars)
    SL.info(f"3D stellar half mass radius: {r_half:.3f}")
    eff_rad = np.mean(
        [pygad.analysis.half_mass_radius(snap.stars, proj=i) for i in range(3)]
    )
    SL.info(f"Stellar effective radius: {eff_rad:.3f}")
    fig_re, ax_re = plt.subplots(1, 1)
    lt = bgs.literature.LiteratureTables.load_sahu_2020_data()
    lt.scatter("logM*_sph", "logRe_maj_kpc", ax=ax_re)
    ax_re.plot(np.log10(sum(snap.stars["mass"])), np.log10(eff_rad), **mkwargs)
    if args.savefig is not None:
        bgs.plotting.savefig(os.path.join(args.savefig, "eff_rad.png"), fig=fig_re)
else:
    SL.warning("No half mass nor effective radius calculated!")
    r_half = None
    eff_rad = None

# create a ball mask for snapshot
if r_half is not None:
    ball_mask = pygad.BallMask(r_half)

# dark matter fraction in R1/2
if "dm" in valid_fams and r_half is not None:
    dm_frac = sum(snap.dm[ball_mask]["mass"]) / sum(snap[ball_mask]["mass"])
    SL.info(f"DM fraction with half mass radius: {dm_frac:.3f}")
    fig_dm, ax_dm = plt.subplots(1, 1)
    lt = bgs.literature.LiteratureTables.load_jin_2020_data()
    lt.scatter("log(M*/Msun)", "f_DM", ax=ax_dm)
    ax_dm.plot(np.log10(sum(snap.stars["mass"])), dm_frac, **mkwargs)
    if args.savefig is not None:
        bgs.plotting.savefig(os.path.join(args.savefig, "fdm.png"), fig=fig_dm)

# some stellar stuff
if "stars" in valid_fams:
    # inner velocity dispersion
    vsig2 = np.full(3, np.nan)
    for i in range(3):
        vsig2[i] = (
            pygad.analysis.los_velocity_dispersion(snap.stars[ball_mask], proj=i) ** 2
        )
    vsig = np.sqrt(np.mean(vsig2))
    SL.info(f"Projected stellar velocity dispersion: {vsig:.3e}")
    fig_msig, ax_msig = plt.subplots(1, 1)
    lt = bgs.literature.LiteratureTables.load_vdBosch_2016_data()
    lt.scatter(
        "logsigma",
        "logBHMass",
        xerr="e_logsigma",
        yerr=["e_logBHMass", "E_logBHMass"],
        ax=ax_msig,
    )
    ax_msig.plot(np.log10(vsig), np.log10(sum(snap.bh["mass"])), **mkwargs)
    if args.savefig is not None:
        bgs.plotting.savefig(os.path.join(args.savefig, "msig.png"), fig=fig_msig)

    # stellar formation time
    if "age" in snap.stars.available_blocks():
        fig_age, ax_age = plt.subplots(1, 1)
        ages = bgs.general.particle_ages(snap.stars)
        ax_age.hist(ages, 50)
        ax_age.set_xlabel(f"Age {ages.units}")
        ax_age.set_ylabel("Count")
        ax_age.set_yscale("log")
        if args.savefig is not None:
            bgs.plotting.savefig(
                os.path.join(args.savefig, "star_formation.png"), fig=fig_age
            )
    else:
        SL.warning("Block 'age' not available!")


plt.show()
