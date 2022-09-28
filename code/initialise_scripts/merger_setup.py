import argparse
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Perturb a particle in a snapshot", allow_abbrev=False)
parser.add_argument(type=str, dest="paramfile", help="path to parameter file")
parser.add_argument(type=str, dest="method", help="set up new, or perturb BH or field particle", choices=["new", "field", "bh"])
args = parser.parse_args()

merger = cmf.initialise.MergerIC(args.paramfile, exist_ok=True)
if args.method == "new":
    merger.setup()
elif args.method == "field":
    merger.perturb_field_particle()
else:
    merger.perturb_bhs()