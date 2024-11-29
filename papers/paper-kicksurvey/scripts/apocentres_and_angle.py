import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


bgs.plotting.check_backend()


parser = argparse.ArgumentParser(
    description="Plot projected density image for 600km/s case",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dist-threshold", dest="threshold", type=float, help="Minimum detection distance threshold", default=5
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbose",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

rng = np.random.default_rng(42)

gp = bgs.analysis.VkickApocentreGP.load_fit(
    "/scratch/pjohanss/arawling/collisionless_merger/stan_files/gp-vkick-apo/kick-apo/gp_analytic-20241127105025*.csv",
    figname_base=f"gaussian_processes/",
    premerger_ketjufile="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output",
    rng=rng
)

analysis_params = bgs.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml")

gp.extract_data(None)

SL.info(f"Number of simulations with usable data: {gp.num_groups}")

if args.verbose == "DEBUG":
    gp.print_obs_summary()

# initialise the data dictionary
gp.set_stan_data()

analysis_params["stan"]["GP_sample_kwargs"]["output_dir"] = os.path.join(
    bgs.DATADIR, f"stan_files/gp-vkick-apo/kick-apo"
)


# run the model
gp.sample_model(
    sample_kwargs=analysis_params["stan"]["GP_sample_kwargs"]
)

# get fraction of apocentres above X kpc
frac_above_X = gp.fraction_apo_above_threshold(args.threshold)
print(f"{frac_above_X*100:.3f}% of sampled apocentres are above {args.threshold:.2f}kpc")
frac_above_X_proj = gp.fraction_apo_above_threshold(args.threshold, proj=True)
print(f"{frac_above_X_proj*100:.3f}% of sampled projected apocentres are above {args.threshold:.2f}kpc")

# make the plots
fig, ax = plt.subplots(1, 2, sharex="all")
fig.set_figwidth(2 * fig.get_figwidth())
hdi_levels = [50, 75, 99]

# plot 1: vkick - r_apo relation from GP
ax[0].set_yscale("log")
ax[0].set_xlabel(gp.input_qtys_labs[0])
ax[0].set_ylabel(gp.folded_qtys_labs[0])
gp.posterior_OOS_plot(
    xmodel="x2", ymodel=gp.folded_qtys_posterior[0], ax=ax[0], smooth=True, save=False, levels=hdi_levels
)
ax[0].axhline(args.threshold, ls=":", lw=1, c="k")
ax[0].text(1200, 0.2 * args.threshold, f"{(1-frac_above_X)*100:.1f}%", va="top")
ax[0].text(1200, 5 * args.threshold, f"{(frac_above_X)*100:.1f}%", va="bottom")
#ax[0].scatter(gp.stan_data["x1"], gp.stan_data["y1"], color=bgs.plotting.mplColours()[1], ec="k", lw="0.5")

# plot 3: vkick - angle offset relation from GP
ax[1].set_xlabel(gp.input_qtys_labs[0])
ax[1].set_ylabel(r"$\theta$")
gp.plot_angle_to_exceed_threshold(args.threshold, levels=hdi_levels, ax=ax[1], save=False)
ax[1].axvspan(-10, 480, hatch="/", color="gray", ec="none", fc="none", alpha=0.4, zorder=0.1)
ax[1].text(150, 45, f"$r_\mathrm{{apo}}<{args.threshold:.1f}\,\mathrm{{kpc}}$", backgroundcolor="w", rotation="vertical")
ax[1].set_xlim(1, np.max(gp.stan_data["x2"]))
ax[1].set_ylim(0, 90)

bgs.plotting.savefig(figure_config.fig_path("apocentres.pdf"), force_ext=True)
