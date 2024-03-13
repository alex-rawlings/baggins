import os.path
import baggins as bgs

parser = bgs.utils.argparse_for_stan(
    "Run Gaussian process method for deflection angle - eccentricity relation."
)
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    action="append",
    default=[],
    dest="extra_dirs",
    help="other directories to compare",
)
args = parser.parse_args()


SL = bgs.setup_logger("script", args.verbose)

if args.type == "new":
    hmq_dirs = []
    hmq_dirs.append(args.dir)
    if args.extra_dirs:
        hmq_dirs.extend(args.extra_dirs)
        SL.debug(f"Directories are: {hmq_dirs}")
else:
    hmq_dirs = None
analysis_params = bgs.utils.read_parameters(args.apf)

figname_base = "gaussian_processes"
model_file = "stan/gp.stan"

if args.type == "loaded":
    GP = bgs.analysis.DeflectionAngleGP.load_fit(
        model_file, args.dir, figname_base, args.NOOS
    )
else:
    GP = bgs.analysis.DeflectionAngleGP(model_file, "", figname_base, args.NOOS)

GP.extract_data(analysis_params, hmq_dirs)
e_ini = f"e0-{GP.e_ini:.2f}".replace(".", "")
GP.figname_base = os.path.join(GP.figname_base, f"{e_ini}/{e_ini}")

GP.set_stan_data()

analysis_params["stan"]["deflection_GP_kwargs"]["output_dir"] = os.path.join(
    bgs.DATADIR, f"stan_files/gps/{e_ini}"
)

GP.sample_model(sample_kwargs=analysis_params["stan"]["deflection_GP_kwargs"])
GP.all_plots()
