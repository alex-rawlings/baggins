import argparse
import os.path
import matplotlib.pyplot as plt
import baggins as bgs


parser = argparse.ArgumentParser(
    description="fit modified NFW density profile with Stan",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="data file", dest="files", type=str)
parser.add_argument("-s", "--save", help="save location", dest="save", type=str)
parser.add_argument(
    "--saveOOS", help="save sampled density data", dest="saveOOS", type=str
)
parser.add_argument(
    "-p", "--prior", help="prior analysis", dest="prior", action="store_true"
)
parser.add_argument(
    "-L", "--loaded", action="store_true", dest="loaded", help="loaded from previous"
)
parser.add_argument(
    "--no-plots", action="store_true", dest="noplots", help="don't make plots"
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

figname_base = "mNFW_density_simple"
if args.loaded:
    mNFW = bgs.analysis.ModifiedNFWModelSimple.load_fit(
        args.files, figname_base=figname_base
    )
else:
    mNFW = bgs.analysis.ModifiedNFWModelSimple(figname_base=figname_base)

mNFW.extract_data(args.files)
sample_kwargs = {"adapt_delta": 0.995, "max_treedepth": 15}
if args.save is not None:
    sample_kwargs["output_dir"] = os.path.join(args.save, mNFW.merger_id)

if args.verbose == "DEBUG":
    mNFW.print_obs_summary()

# initialise the data dictionary
mNFW.set_stan_data(rmin=1e-2)
if args.prior:
    mNFW.sample_prior(sample_kwargs=sample_kwargs)
    mNFW.all_prior_plots()
else:
    mNFW.sample_model(
        sample_kwargs=sample_kwargs, diagnose=False
    )  # TODO change diagnose

    if not args.noplots:
        mNFW.all_posterior_pred_plots()

        # set up guiding Plummer lines
        fig, ax = plt.subplots()
        mNFW.add_data_to_predictive_plot(ax=ax, xobs="r", yobs="density")
        for g, ls in zip((0.5, 1, 1.5), ("-", ":", "--")):
            mNFW.add_guiding_NFW(ax=ax, rS=0.2, N=3)
        mNFW.all_posterior_OOS_plots(ax=ax)
mNFW.print_parameter_percentiles(mNFW.latent_qtys)

if args.saveOOS is not None:
    mNFW.save_density_data_to_npz(args.saveOOS)
