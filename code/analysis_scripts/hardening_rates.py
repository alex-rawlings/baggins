import argparse
import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Compare the analytical hardening rate to ketju output", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="path")
parser.add_argument(type=str, help="perturbation number", dest="num")
parser.add_argument("-r", "--radiusgw", type=float, help="Radius [pc] above which GW emission expected to be negligible", dest="rgw", default=15)
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.path, verbose=False)

# create the bh binary class that will hold all the data
bh_binary = cmf.analysis.BHBinary(args.path, args.num, args.rgw)
bh_binary.print()
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios":[3,1]})
bh_binary.plot(ax)
plt.show()