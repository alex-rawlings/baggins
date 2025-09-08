import argparse
import os.path
import baggins as bgs


parser = argparse.ArgumentParser(
    description="fit ABG density profile with Stan",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="data file", dest="files", type=str)
# parser.add_argument(help="ID", dest="ID", type=str)
parser.add_argument("-s", "--save", help="save location", dest="save", type=str)
parser.add_argument(
    "--saveOOS", help="save sampled density data", dest="saveOOS", type=str
)
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

abgdens = bgs.analysis.ABGDensityModelSimple("abg_density")
abgdens.read_data_from_txt(args.files, skiprows=1)
sample_kwargs = {"adapt_delta": 0.995, "max_treedepth": 15}
if args.save is not None:
    sample_kwargs["output_dir"] = os.path.join(args.save, abgdens.merger_id)

if args.verbose == "DEBUG":
    abgdens.print_obs_summary()

# initialise the data dictionary
abgdens.set_stan_data()
if args.prior:
    abgdens.sample_prior(sample_kwargs=sample_kwargs)

    abgdens.all_prior_plots()
else:
    abgdens.sample_model(sample_kwargs=sample_kwargs)

    abgdens.all_posterior_pred_plots()
    abgdens.all_posterior_OOS_plots()
abgdens.print_parameter_percentiles(abgdens.latent_qtys)

if args.saveOOS is not None:
    abgdens.save_density_data_to_npz(args.saveOOS)
