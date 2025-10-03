import argparse
import os.path
import matplotlib.pyplot as plt
import baggins as bgs


parser = argparse.ArgumentParser(
    description="fit ABG density profile with Stan",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(help="data file", dest="files", type=str)
parser.add_argument(
    "-m",
    "--model",
    help="simple or hierarchical model",
    choices=["s", "h"],
    default="s",
    type=str,
    dest="model",
)
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

if args.model == "s":
    abgdens = bgs.analysis.ABGDensityModelSimple("abg_density_simple")
else:
    abgdens = bgs.analysis.ABGDensityModelHierarchy("abg_density_hierarchy")
abgdens.extract_data(args.files, skiprows=1)
sample_kwargs = {"adapt_delta": 0.995, "max_treedepth": 15}
if args.save is not None:
    sample_kwargs["output_dir"] = os.path.join(args.save, abgdens.merger_id)

if args.verbose == "DEBUG":
    abgdens.print_obs_summary()

# initialise the data dictionary
abgdens.set_stan_data(rmin=1e-2)
if args.prior:
    abgdens.sample_prior(sample_kwargs=sample_kwargs)

    abgdens.all_prior_plots()
else:
    abgdens.sample_model(sample_kwargs=sample_kwargs)

    abgdens.all_posterior_pred_plots()

    # set up guiding Plummer lines
    fig, ax = plt.subplots()
    abgdens.add_data_to_predictive_plot(ax=ax, xobs="r", yobs="density")
    abgdens.add_guiding_Plummer(ax=ax, rS=0.2)
    bgs.plotting.add_log_guiding_gradients(
        ax=ax, x0=0.085, x1=0.2, y1=1e3, b=[-2, -1, 0, 1, 2], offset=-0.01
    )
    abgdens.all_posterior_OOS_plots(ax=ax)
abgdens.print_parameter_percentiles(abgdens.latent_qtys)

if args.saveOOS is not None:
    abgdens.save_density_data_to_npz(args.saveOOS)
