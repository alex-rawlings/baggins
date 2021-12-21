import argparse
import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf

parser = argparse.ArgumentParser(description="Compare the analytical hardening rate to ketju output", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="path")
parser.add_argument(type=str, help="perturbation number", dest="num")
parser.add_argument("-r", "--radiusgw", type=float, help="Radius [pc] above which GW emission expected to be negligible", dest="rgw", default=15)
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.path)
data_path = os.path.join(pfv.full_save_location, pfv.perturbSubDir, args.num, "output")

bhfile = cmf.utils.get_ketjubhs_in_dir(data_path)[0]
snaplist = cmf.utils.get_snapshots_in_dir(data_path)

time_offset = pfv.perturbTime * 1000 #in Myr
print("Perturbation applied at {:.1f} Myr".format(time_offset))

# create the bh binary class that will hold all the data
bh_binary = cmf.analysis.BHBinary(bhfile, snaplist, args.rgw)
bh_binary.get_influence_and_hard_radius()
bh_binary.fit_analytic_form()
bh_binary.time_estimates()
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios":[3,1]})
bh_binary.plot(ax, time_offset)
plt.show()