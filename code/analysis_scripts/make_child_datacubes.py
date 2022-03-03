import argparse
import os.path
import cm_functions as cmf

#set up command line arguments
parser = argparse.ArgumentParser(description="Create datacubes of simulation perturbed runs.", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="pf")
parser.add_argument(type=str, help="perturbation number", dest="pnum")
parser.add_argument("-r", "--radiusgw", type=float, help="Radius [pc] above which GW emission expected to be negligible", dest="rgw", default=15)
parser.add_argument("-l", "--location", type=str, help="Location of saved file", dest="saveloc", default="/scratch/pjohanss/arawling/collisionless_merger/mergers/cubes")
parser.add_argument("-v", "--verbose", help="verbose printing", dest="verbose", action="store_true")
args = parser.parse_args()


if args.pnum == "all":
    # TODO make this more general, what if there aren't 10 perturbations?
    perturb_id = [str(i) for i in range(10)]
else:
    perturb_id = [args.pnum]

for pid in perturb_id:
    dc = cmf.analysis.ChildSim(args.pf, pid, verbose=args.verbose)
    file_save_name = os.path.join(args.saveloc, "cube-{}.hdf5".format(dc.merger_name))
    dc.make_hdf5(file_save_name)
