import os.path
import numpy as np
import baggins as bgs


bgs.plotting.check_backend()

parser = bgs.utils.argparse_for_stan(
    "Run stan model for Core-Sersic model",
    pos_args=[
        {"dest":"ketju", "help":"premerger ketju file", "type":str}
    ])
parser.add_argument(
    "-s", "--sample", dest="sample", help="observation sample", default="misc"
)
parser.add_argument(
    "--maxvel", type=float, dest="maxvel", help="maximum velocity to fit to"
)
parser.add_argument(
    "--dist-threshold", dest="threshold", type=float, help="Minimum detection distance threshold", default=5
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

full_figsize = bgs.plotting.get_figure_size(
    args.publish, full=True, multiplier=[1.3, 1.3]
)

figname_base = f"gaussian_processes/{args.sample}/vkick_apo"
rng = np.random.default_rng(42)

if args.type == "new":
    hmq_dir = args.dir
    SL.debug(f"Input data read from {hmq_dir}")
    gp = bgs.analysis.VkickApocentreGP(
        figname_base=figname_base,
        premerger_ketjufile=args.ketju,
        rng=rng
    )
else:
    hmq_dir = None
    gp = bgs.analysis.VkickApocentreGP.load_fit(
        args.dir,
        figname_base=figname_base,
        premerger_ketjufile=args.ketju,
        rng=rng
    )
analysis_params = bgs.utils.read_parameters(args.apf)

gp.extract_data(hmq_dir, maxvel=args.maxvel)

SL.info(f"Number of simulations with usable data: {gp.num_groups}")

if args.verbose == "DEBUG":
    gp.print_obs_summary()

# initialise the data dictionary
gp.set_stan_data()

analysis_params["stan"]["GP_sample_kwargs"]["output_dir"] = os.path.join(
    bgs.DATADIR, f"stan_files/gp-vkick-apo/{args.sample}"
)


# run the model
gp.sample_model(
    sample_kwargs=analysis_params["stan"]["GP_sample_kwargs"]
)

ax = gp.all_plots(full_figsize)
gp.print_parameter_percentiles(gp.latent_qtys)

# get fraction of apocentres above X kpc
frac_above_X = gp.fraction_apo_above_threshold(args.threshold)
print(f"{frac_above_X*100:.3f}% of sampled apocentres are above {args.threshold:.2f}kpc")
