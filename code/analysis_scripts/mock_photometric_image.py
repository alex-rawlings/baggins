"""
mock_photometric_image.py

Driver script for generating mock photometric images using
baggins.analysis.instruments.PhotometricInstrument subclasses, which wrap
Synthesizer's particle imaging pipeline
(https://synthesizer-project.github.io/synthesizer/observables/imaging/particle_imaging.html).
"""

import argparse
import pygad
import baggins as bgs

parser = argparse.ArgumentParser(
    description="Generate mock photometric images",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(dest="snapfile", type=str, help="snapshot to generate image of")
parser.add_argument("-z", dest="redshift", type=float, help="redshift", default=0.6)
parser.add_argument(
    "-i",
    "--instrument",
    dest="instrument",
    type=str,
    help="instrument",
    choices=["HST", "Euclid"],
    default="Euclid",
)
args = parser.parse_args()

snapfile = "/orion/ptmp/arawling/recoil-sims/k0540_snap_009.hdf5"
redshift = 0.6


snap = pygad.Snapshot(args.snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)
# snap = snap[pygad.BallMask(10)]

if args.instrument == "Euclid":
    instr = bgs.analysis.Euclid_VIS(z=args.redshift)
else:
    instr = bgs.analysis.HSTWFC3(z=args.redshift)
# This snapshot only tracks a 2 kpc nuclear region around the BH -- cap
# the instrument's default 40 kpc frame down to something comparable to
# the data, so the galaxy isn't a tiny speck in an otherwise empty image.
# instr.max_extent = 10.0
# This snapshot has no age/metallicity blocks (collisionless-only
# output), so every star is assigned the same SSP -- expect a
# structurally realistic but colour-flat mock from this dataset.
# `softening` is left as None so the instrument derives a smoothing
# length from its own resolution element instead of an arbitrary value.
instr.load_and_project_galaxy(snap, ages=1e9, metallicity=0.03396)
instr.generate_particle_spectra()
instr.build_instrument()
instr.observe()
bgs.plotting.savefig("test.png")
