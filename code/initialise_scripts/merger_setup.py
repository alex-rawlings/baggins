import argparse
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Set up or perturb a merger system", allow_abbrev=False)
parser.add_argument(type=str, dest="paramfile", help="path to parameter file")
parser.add_argument(type=str, dest="method", help="set up new, or perturb BH or field particle", choices=["new", "field", "bh"])
parser.add_argument("-o", "--overwrite", dest="overwrite", help="allow overwriting of files?", action="store_true")
args = parser.parse_args()

merger = cmf.initialise.MergerIC(args.paramfile, exist_ok=args.overwrite)
if args.method == "new":
    merger.setup()
elif args.method == "field":
    merger.perturb_field_particle()
else:
    merger.perturb_bhs()