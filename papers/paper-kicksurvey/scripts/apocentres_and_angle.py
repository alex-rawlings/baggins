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
    "-n", "--new", help="run new Stan sampling", action="store_true", dest="new"
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

gp_kwargs = dict(
    figname_base="gaussian_processes/",
    premerger_ketjufile="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output",
    rng=rng,
)
if args.new:
    gp = bgs.analysis.VkickApocentreGP(**gp_kwargs)
    data_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data"
else:
    # TODO make this dynamic, choose last sampled model
    gp = bgs.analysis.VkickApocentreGP.load_fit(
        "/scratch/pjohanss/arawling/collisionless_merger/stan_files/gp-vkick-apo/kick-apo/gp_analytic-20250127173129*.csv",
        **gp_kwargs,
    )
    data_dir = None

# XXX this has to be determined prior to calling this routine
core_sig = 270


def threshold_dist(x, core_sig=core_sig):
    y = 1.56e-2 * x - 3.63
    y[x < core_sig] = 1000
    return y


analysis_params = bgs.utils.read_parameters(
    "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"
)

gp.extract_data(d=data_dir, maxvel=1320)

SL.info(f"Number of simulations with usable data: {gp.num_groups}")

if args.verbose == "DEBUG":
    gp.print_obs_summary()
    for v, ra in zip(gp.obs_collapsed["vkick"], gp.obs_collapsed["rapo"]):
        print(f"Kick {v:.1e}: {ra:.1e} kpc")

# initialise the data dictionary
gp.set_stan_data()

analysis_params["stan"]["GP_sample_kwargs"]["output_dir"] = os.path.join(
    bgs.DATADIR, "stan_files/gp-vkick-apo/kick-apo"
)


# run the model
gp.sample_model(sample_kwargs=analysis_params["stan"]["GP_sample_kwargs"])

# get fraction of apocentres above X kpc
frac_above_X = gp.fraction_apo_above_threshold(threshold_dist)
print(f"{frac_above_X*100:.3f}% of sampled apocentres are above the threshold function")
frac_above_X_proj = gp.fraction_apo_above_threshold(threshold_dist, proj=True)
print(
    f"{frac_above_X_proj*100:.3f}% of sampled projected apocentres are above the threshold function"
)

# make the plots
fig, ax = plt.subplots(1, 3, sharex="all")
fig.set_figwidth(2.5 * fig.get_figwidth())
hdi_levels = [50, 75, 99]

# plot 1: vkick - r_apo relation from GP
ax[0].set_yscale("log")
ax[0].set_xlabel(gp.input_qtys_labs[0])
ax[0].set_ylabel(gp.folded_qtys_labs[0])
gp.posterior_OOS_plot(
    xmodel="x2",
    ymodel=gp.folded_qtys_posterior[0],
    ax=ax[0],
    smooth=True,
    save=False,
    levels=hdi_levels,
)

ylims = ax[0].get_ylim()
vk_threshold = np.linspace(core_sig + 1, np.max(gp.stan_data["x2"]), 100)
ax[0].plot(vk_threshold, threshold_dist(vk_threshold), ls=":", lw=1, c="k", zorder=2)
ax[0].text(1000, 6, f"{(1-frac_above_X)*100:.1f}%", va="top")
ax[0].text(1000, 25, f"{(frac_above_X)*100:.1f}%", va="bottom")
ax[0].set_ylim(0.1, ylims[1])

# plot 2: vkick - angle offset relation from GP
ax[1].set_xlabel(gp.input_qtys_labs[0])
ax[1].set_ylabel(r"$\theta_\mathrm{min}$")
gp.plot_angle_to_exceed_threshold(
    threshold_dist,
    levels=hdi_levels,
    ax=ax[1],
    save=False,
    smooth_kwargs={"mode": "nearest", "window_length": 5},
)
ax[1].set_xlim(0.9 * core_sig, np.max(gp.stan_data["x2"]))
ax[1].set_ylim(0, 90)
ax[1].text(1000, 60, r"$\mathrm{Detectable}$")
ax[1].text(400, 15, r"$\mathrm{Not\; detectable}$")

# plot 3: probability of distribution of observable vkicks
bin_width = 100
bins = np.arange(
    np.nanmin(gp.stan_data["x2"]), np.nanmax(gp.stan_data["x2"]) + bin_width, bin_width
)
gp.plot_observable_fraction(threshold_dist, bins=bins, ax=ax[2], save=False)
ax[2].set_xlabel(gp.input_qtys_labs[0])
ax[2].set_ylabel(r"$f(v_\mathrm{kick})$")
ax[2].legend()

bgs.plotting.savefig(figure_config.fig_path("apocentres.pdf"), force_ext=True)
