import argparse
import baggins as bgs

parser = argparse.ArgumentParser(description="Fit an analytical potential", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(type=str, dest="snapfile", help="path to snapshot(s)")
parser.add_argument("--extent", dest="extent", type=float, help="extent to fit potential within", default=1.0)
parser.add_argument("--majoraxis", dest="majoraxis", type=str, help="major merger axis", default="x", choices=["x", "y"])
args = parser.parse_args()

PotentialConstructor = bgs.analysis.PotentialFitter(
    snapfile=args.snapfile,
    major_axis=args.majoraxis
)
PotentialConstructor.fit_potential(extent=args.extent)
print(PotentialConstructor)
PotentialConstructor.plot()
bgs.plotting.savefig("pot.png", save_kwargs={"dpi":500})
PotentialConstructor.plot_potential_1D()
bgs.plotting.savefig("pot_1D.png", save_kwargs={"dpi":500})