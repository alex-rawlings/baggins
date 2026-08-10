import argparse
import os
import baggins as bgs

parser = argparse.ArgumentParser(
    description="Sinking of BH in Plummer Sphere",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--sim", dest="simdir", type=str, help="simulation output directory", nargs="+"
)
parser.add_argument("--mbh", dest="mbh", type=float, help="bh mass", default=3e7)
parser.add_argument(
    "--mstar", dest="mstar", type=float, help="stellar mass", default=1e11
)
parser.add_argument(
    "--scale-rad", dest="scalerad", type=float, help="scale radius", default=1.5
)
args = parser.parse_args()


kwargs = {"MBH": args.mbh, "Mstar": args.mstar, "a": args.scalerad}
ap = bgs.literature.SinkingBHPlummer(**kwargs)
ap.evolve()
ax = ap.plot()
ap.plot_bmax_range(ax, bmax_fac_range=(1, 3))
if args.simdir is not None:
    for simdir in args.simdir:
        ap.plot_simulation(simdir, ax=ax, label="Sim")
figname = os.path.join(bgs.FIGDIR, "plummer_sink/plummer_sink.png")
os.makedirs(os.path.dirname(figname), exist_ok=True)
bgs.plotting.savefig(figname)
