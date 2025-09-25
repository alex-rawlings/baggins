import argparse
import os
import numpy as np
import baggins as bgs


parser = argparse.ArgumentParser(
    description="fit ABG density profile with Stan",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="data file", dest="files", type=str)
parser.add_argument(
    "--figname", help="figure name", dest="figname", type=str, default="gp_general"
)
parser.add_argument("--seed", help="random seed", dest="seed", type=int, default="42")
parser.add_argument(
    "--thin", help="thin observations", dest="thin", type=int, default=None
)
parser.add_argument(
    "--logx", action="store_true", help="fit x data in log10 space", dest="logx"
)
parser.add_argument(
    "--logy", action="store_true", help="fit y data in log10 space", dest="logy"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

figdir = os.path.join(bgs.FIGDIR, "gaussian_processes/gp_general")
os.makedirs(figdir, exist_ok=True)
figname = os.path.join(figdir, args.figname)

rng = np.random.default_rng(args.seed)

gp = bgs.analysis.GeneralGP(figname_base=figname, rng=rng)
SL.debug("Extracting data")
gp.extract_data(args.files, logx=args.logx, logy=args.logy)
if args.thin is not None:
    gp.thin_observations(args.thin)
SL.debug("Setting stan data")
gp.set_stan_data()
SL.debug("Begin sampling model")
gp.sample_model()
SL.debug("Model sampling finished")
gp.all_plots()
