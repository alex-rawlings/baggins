import argparse
import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Compare the analytical hardening rate to ketju output", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="path")
parser.add_argument(type=str, help="perturbation number", dest="num")
parser.add_argument("-a", "--aparams", type=str, help="path to analysis parameter file", dest="apf", default=os.path.join(cmf.HOME, "projects/collisionless-merger-sample/parameters/parameters-analysis/datacubes.py"))
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.path, verbose=False)

# create the bh binary class that will hold all the data
bh_binary = cmf.analysis.BHBinary(args.path, args.num, args.apf)
bh_binary.print()
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios":[3,1]})
bh_binary.plot(ax)
plt.show()