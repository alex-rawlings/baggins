import argparse
import baggins as bgs

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Fit a Dehnen profile with stan",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="snapshot to analyse", dest="file", type=str)
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

dehnen = bgs.analysis.DehnenModel("dehnen_fits/dehnen")
dehnen.extract_data(args.file)


if args.verbose == "DEBUG":
    dehnen.print_obs_summary()

# initialise the data dictionary
dehnen.set_stan_data()

dehnen.sample_model()

dehnen.all_posterior_pred_plots()
dehnen.all_posterior_OOS_plots()
dehnen.print_parameter_percentiles(dehnen.latent_qtys)
