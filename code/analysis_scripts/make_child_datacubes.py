import argparse
import cm_functions as cmf

#set up command line arguments
parser = argparse.ArgumentParser(description="Create datacubes of simulation perturbed runs.", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="pf")
parser.add_argument(type=str, help="perturbation number", dest="pnum")
parser.add_argument("-r", "--radiusgw", type=float, help="Radius [pc] above which GW emission expected to be negligible", dest="rgw", default=15)
parser.add_argument("-N", "--name", type=str, help="Name of saved file", dest="fname", default=None)
args = parser.parse_args()


dc = cmf.analysis.ChildSim(args.pf, args.pnum)

if args.fname is None:
    dc.make_hdf5(dc.merger_name + ".hdf5")
else:
    if args.fname.endswith(".hdf5"):
        args.fname += ".hdf5"
    dc.make_hdf5(args.fname)
