import argparse
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Check isolated system stability",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(type=str, help="path to snapshot", dest="path")
parser.add_argument(
    "--min-part-count",
    type=int,
    help="minimum particle count per bin for beta",
    dest="min",
    default=1000,
)
parser.add_argument(
    "--stride", type=int, help="use every ith snapshot", dest="stride", default=None
)
parser.add_argument(
    "-f",
    "--family",
    help="particle family",
    choices=["stars", "dm"],
    default="stars",
    dest="fam",
)
parser.add_argument(
    "--mass-fracs",
    nargs="+",
    type=float,
    help="mass fractions",
    dest="mass_fracs",
    default=None,
)
parser.add_argument(
    "--num-bins",
    type=int,
    help="number radial bins for beta",
    dest="nbins",
    default=10,
)
parser.add_argument(
    "--rmax",
    type=float,
    help="maximum radius for beta",
    dest="rmax",
    default=1e5,
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()


SL = bgs.setup_logger("script", args.verbose)


# mass fractions for Lagrangian radii
if args.mass_fracs is None:
    args.mass_fracs = [0.1, 0.25, 0.5, 0.7, 0.9]
args.mass_fracs.sort()

fig, ax = plt.subplots(1, 2)
fig.set_figwidth(1.5 * fig.get_figwidth())

ax[0].set_xlabel(r"$t/\mathrm{Gyr}$")
ax[0].set_ylabel(r"$R_\mathrm{Lang.}/\mathrm{kpc}$")
ax[1].set_xlabel(r"$r/\mathrm{kpc}$")
ax[1].set_ylabel(r"$\beta$")


snap_gen = bgs.analysis.SnapshotIterator(args.path, stride=args.stride)

# set up arrays to hold data
t = np.full(snap_gen.len, np.nan)
lang_radii = np.full((snap_gen.len, len(args.mass_fracs)), np.nan)

cmapperR, smR = bgs.plotting.create_normed_colours(
    min(args.mass_fracs), max(args.mass_fracs), cmap="crest_r"
)
cmappert, smt = bgs.plotting.create_normed_colours(0, snap_gen.len)

for i, _t, snap in snap_gen.make_generator():
    if args.fam == "stars":
        snap = snap.stars
    else:
        snap = snap.dm

    for j, mf in enumerate(args.mass_fracs):
        lang_radii[i, j] = bgs.analysis.lagrangian_radius(snap, mass_frac=mf)
    t[i] = _t

    # ensure bins of equal particle number for the first nbin bins and one
    # final bin to rmax
    N_per_bin = int(len(snap[pygad.BallMask(args.rmax)]) / args.nbins)
    r_edges = bgs.mathematics.equal_count_bins(
        snap[pygad.BallMask(args.rmax)]["r"], N_per_bin
    )
    beta, bincounts = bgs.analysis.velocity_anisotropy(snap, r_edges=r_edges)
    r_centres = bgs.mathematics.get_histogram_bin_centres(r_edges)
    ax[1].semilogx(r_centres, beta, c=cmappert(i))

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

# plot lagrangian radii
for i in range(lang_radii.shape[-1]):
    ax[0].plot(t, lang_radii[..., i], c=cmapperR(args.mass_fracs[i]))
    SL.debug(list(map(lambda x: f"{x:.2e}", lang_radii[..., i])))

ax[0].set_yscale("log")
# let's keep the beta axis sensible
ax[1].set_ylim(-2, 1)

plt.colorbar(smR, ax=ax[0], label="mass frac")
plt.colorbar(smt, ax=ax[1], label="Snapshot")
fig.suptitle(f"{args.fam}: {args.path}")

os.makedirs(os.path.join(bgs.FIGDIR, "stability"), exist_ok=True)
bgs.plotting.savefig(
    os.path.join(
        bgs.FIGDIR,
        "stability",
        f"stability_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
    )
)
