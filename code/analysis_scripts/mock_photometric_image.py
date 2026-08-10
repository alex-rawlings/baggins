"""
mock_photometric_image.py

Driver script for generating mock photometric images using
baggins.analysis.instruments.PhotometricInstrument subclasses, which wrap
Synthesizer's particle imaging pipeline
(https://synthesizer-project.github.io/synthesizer/observables/imaging/particle_imaging.html).
"""

import argparse
import os.path
import matplotlib.pyplot as plt
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
    choices=["HST", "Euclid", "JWST-MIRI", "JWST-NIRCam", "FORS2"],
    default="Euclid",
)
parser.add_argument(
    "--rgb",
    dest="rgb",
    type=int,
    nargs="*",
    help="filters to use for RGB image",
    default=[0, 1, 2],
)
args = parser.parse_args()

snapfile = "/orion/ptmp/arawling/recoil-sims/k0540_snap_009.hdf5"

snap = pygad.Snapshot(args.snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)

rgb = {"r": None, "g": None, "b": None}

if args.instrument == "Euclid":
    instr = bgs.analysis.Euclid_VIS(z=args.redshift)
elif args.instrument == "HST":
    instr = bgs.analysis.HSTWFC3(z=args.redshift)
    rgb.update(zip("rgb", (bgs.analysis.HST_FILTER_CODES[c] for c in args.rgb)))
elif args.instrument == "JWST-MIRI":
    instr = bgs.analysis.JWST_MIRI(z=args.redshift)
    rgb.update(zip("rgb", (bgs.analysis.JWST_MIRI_FILTER_CODES[c] for c in args.rgb)))
elif args.instrument == "JWST-NIRCam":
    instr = bgs.analysis.JWST_NIRCam(z=args.redshift)
    rgb.update(zip("rgb", (bgs.analysis.JWST_NIRCam_FILTER_CODES[c] for c in args.rgb)))
elif args.instrument == "FORS2":
    instr = bgs.analysis.VLT_FORS2(z=args.redshift)
    rgb.update(zip("rgb", (bgs.analysis.VLT_FORS2_FILTER_CODES[c] for c in args.rgb)))

instr.load_and_project_galaxy(snap, ages=1e9, metallicity=0.03396)
instr.generate_particle_spectra()
instr.build_instrument()
instr.observe()
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"mock_obs/{instr.name}_flux.png"))
plt.close()
try:
    instr.make_rgb_image(**rgb)
    bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"mock_obs/{instr.name}_rgb.png"))
    plt.close()
except ValueError:
    print(f"Can't make RGB image for instrument {instr.name}")
