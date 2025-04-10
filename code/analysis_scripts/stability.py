import argparse
import os
from datetime import datetime
from tqdm import tqdm
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
    "-m",
    "--min",
    type=int,
    help="minimum particle count per bin for beta",
    dest="min",
    default=None,
)
parser.add_argument(
    "--stride", type=int, help="use every ith snapshot", dest="stride", default=None
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
mass_fracs = [0.1, 0.25, 0.5, 0.7, 0.9]

# radial bin edges for beta profile
r_edges = np.geomspace(1e-2, 30, 10)
r_centres = bgs.mathematics.get_histogram_bin_centres(r_edges)


fig, ax = plt.subplots(1, 2)
fig.set_figwidth(1.5 * fig.get_figwidth())

ax[0].set_xlabel(r"$t/\mathrm{Gyr}$")
ax[0].set_ylabel(r"$R_\mathrm{Lang.}/\mathrm{kpc}$")
ax[1].set_xlabel(r"$r/\mathrm{kpc}$")
ax[1].set_ylabel(r"$\beta$")

snapfiles = bgs.utils.get_snapshots_in_dir(args.path)
if args.stride is not None:
    snapfiles = snapfiles[:: args.stride]

# set up arrays to hold data
t = np.full(len(snapfiles), np.nan)
lang_radii = np.full((len(snapfiles), len(mass_fracs)), np.nan)

cmapperR, smR = bgs.plotting.create_normed_colours(
    min(mass_fracs), max(mass_fracs), cmap="crest_r"
)
cmappert, smt = bgs.plotting.create_normed_colours(0, len(snapfiles))

for i, snapfile in tqdm(
    enumerate(snapfiles), total=len(snapfiles), desc="Analysing snapshots"
):
    snap = pygad.Snapshot(snapfile, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)

    for j, mf in enumerate(mass_fracs):
        lang_radii[i, j] = bgs.analysis.lagrangian_radius(snap, mass_frac=mf)
    t[i] = bgs.general.convert_gadget_time(snap)

    beta, bincounts = bgs.analysis.velocity_anisotropy(snap, r_edges=r_edges)
    if args.min is not None:
        mask = bincounts > args.min
        ax[1].semilogx(r_centres[mask], beta[mask], c=cmappert(i))
    else:
        ax[1].semilogx(r_centres, beta, c=cmappert(i))

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

# plot lagrangian radii
for i in range(lang_radii.shape[-1]):
    ax[0].plot(t, lang_radii[..., i], c=cmapperR(mass_fracs[i]))

ax[0].set_yscale("log")
# let's keep the beta axis sensible
ax[1].set_ylim(-2, 1)

plt.colorbar(smR, ax=ax[0], label="mass frac")
plt.colorbar(smt, ax=ax[1], label="Snapshot")

os.makedirs(os.path.join(bgs.FIGDIR, "stability"), exist_ok=True)
bgs.plotting.savefig(
    os.path.join(
        bgs.FIGDIR,
        "stability",
        f"stability_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
    )
)
