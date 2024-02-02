import argparse
import os.path
import matplotlib.pyplot as plt
import cm_functions as cmf
import figure_config


parser = argparse.ArgumentParser(description="Plot triaxiality of merger remnants given triaxial data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(dest="path", type=str, help="path to data")
parser.add_argument("-c", "--combine", action="store_true", dest="combine", help="combine datasets")
parser.add_argument("-v", "--verbosity", type=str, default="INFO", choices=cmf.VERBOSITY, dest="verbosity", help="set verbosity level")
args = parser.parse_args()

SL = cmf.setup_logger("script", args.verbosity)

if args.combine:
    pickle_files = cmf.utils.get_files_in_dir(args.path, ext=".pickle")
    pickle_files = [p for p in pickle_files if "triax_v" in p]
    try:
        assert pickle_files
    except AssertionError:
        SL.exception("No pickle files found with correct string pattern!", exc_info=True)
        raise
    data = {}
    for p in pickle_files:
        k = os.path.splitext(os.path.basename(p))[0].replace("triax_", "")
        data[k] = cmf.utils.load_data(p)
    cmf.utils.save_data(data, os.path.join(args.path, "triax_core-study.pickle"))
else:
    assert os.path.isfile(args.path)
    data = cmf.utils.load_data(args.path)

# set up figure
fig, ax = plt.subplots(2,1,sharex="all", sharey="all")
cmapper, sm = cmf.plotting.create_normed_colours(vmin=0, vmax=900, cmap="custom_Blues")

for k, v in data.items():
    if k == "__githash" or k == "__script": continue
    vv = v["ratios"]
    c = cmapper(float(k[1:]))
    ax[0].plot(vv["r"][0], vv["ba"][0], c=c, ls="-")
    ax[1].plot(vv["r"][0], vv["ca"][0], c=c, ls="-")

# add colour bar and other labels
plt.colorbar(sm, ax=ax.ravel().tolist(), label=r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$")
for axi, lab in zip(ax, (r"$b/a$", r"$c/a$")):
    axi.set_xscale("log")
    axi.set_ylabel(lab)
ax[1].set_xlabel(r"$R/\mathrm{kpc}$")

cmf.plotting.savefig(figure_config.fig_path("triaxiality.pdf"), force_ext=True)
plt.show()