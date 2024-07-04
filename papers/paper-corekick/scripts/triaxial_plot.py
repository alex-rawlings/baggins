import argparse
import os.path
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


parser = argparse.ArgumentParser(
    description="Plot triaxiality of merger remnants given triaxial data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(dest="path", type=str, help="path to data")
parser.add_argument(
    "-c", "--combine", action="store_true", dest="combine", help="combine datasets"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)

bgs.plotting.check_backend()

if args.combine:
    pickle_files = bgs.utils.get_files_in_dir(args.path, ext=".pickle")
    pickle_files = [p for p in pickle_files if "triax_v" in p]
    try:
        assert pickle_files
    except AssertionError:
        SL.exception(
            "No pickle files found with correct string pattern!", exc_info=True
        )
        raise
    data = {}
    for p in pickle_files:
        k = os.path.splitext(os.path.basename(p))[0].replace("triax_", "")
        data[k] = bgs.utils.load_data(p)
    bgs.utils.save_data(data, os.path.join(args.path, "triax_core-study.pickle"))
else:
    assert os.path.isfile(args.path)
    data = bgs.utils.load_data(args.path)

# set up figure
fig, ax = plt.subplot_mosaic(
    """
    A
    B
    C
    C
    """,
    sharex=True,
)
fig.set_figheight(fig.get_figheight() * 1.5)
ax["A"].sharey(ax["B"])
vkcols = figure_config.VkickColourMap()

for k, v in data.items():
    if k == "__githash" or k == "__script":
        continue
    vv = v["ratios"]
    c = vkcols.get_colour(float(k[1:]))
    ax["A"].plot(vv["r"][0], vv["ba"][0], c=c, ls="-")
    ax["B"].plot(vv["r"][0], vv["ca"][0], c=c, ls="-")
    ax["C"].plot(
        vv["r"][0], (1 - vv["ba"][0] ** 2) / (1 - vv["ca"][0] ** 2), c=c, ls="-"
    )

# add colour bar and other labels
vkcols.make_cbar(list(ax.values()), extend=None)

for k, lab in zip("ABC", (r"$b/a$", r"$c/a$", r"$T$")):
    ax[k].set_xscale("log")
    ax[k].set_ylabel(lab)
ax["C"].set_xlabel(r"$r/\mathrm{kpc}$")

# add some text for the T plot
ax["C"].set_ylim(0, 1)
ax["C"].axhline(0.5, c="gray", alpha=0.4, ls=":")
ax["C"].text(7, 0.52, "prolate")
ax["C"].text(7, 0.25, "oblate")

bgs.plotting.savefig(figure_config.fig_path("triaxiality.pdf"), force_ext=True)
plt.show()
