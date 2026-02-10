import argparse
import baggins as bgs


parser = argparse.ArgumentParser(
    description="fit Terzic profile with Stan",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="data file", dest="files", type=str)
parser.add_argument(help="ID", dest="ID", type=str)
parser.add_argument(
    "-p", "--prior", help="prior analysis", dest="prior", action="store_true"
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
sample_kwargs = {"adapt_delta": 0.995, "max_treedepth": 15}

terzic = bgs.analysis.TerzicModelSimple("terzic")
terzic.read_data_from_txt(args.files, mergerid=args.ID, skiprows=1)

if args.verbose == "DEBUG":
    terzic.print_obs_summary()

# initialise the data dictionary
terzic.set_stan_data()
if args.prior:
    terzic.sample_prior(sample_kwargs=sample_kwargs)

    terzic.all_prior_plots()
else:
    terzic.sample_model(sample_kwargs=sample_kwargs)

    terzic.all_posterior_pred_plots()
    terzic.all_posterior_OOS_plots()
terzic.print_parameter_percentiles(terzic.latent_qtys)
