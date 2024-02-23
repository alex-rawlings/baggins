import argparse
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf


parser = argparse.ArgumentParser(
    description="Run Stan model for Quinlan hardening parameter", allow_abbrev=False
)
parser.add_argument(type=str, help="directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument(
    "-m",
    "--model",
    help="model to run",
    type=str,
    choices=["simple", "hierarchy"],
    dest="model",
    default="hierarchy",
)
parser.add_argument(
    "-p", "--prior", help="plot for prior", action="store_true", dest="prior"
)
parser.add_argument(
    "-l",
    "--load",
    type=str,
    help="load previous stan file",
    dest="load_file",
    default=None,
)
parser.add_argument(
    "-t", "--thin", type=int, help="thin data", dest="thin", default=100
)
parser.add_argument(
    "-s",
    "--sample",
    help="sample set",
    type=str,
    dest="sample",
    choices=["mcs", "perturb"],
    default="mcs",
)
parser.add_argument(
    "-P", "--Publish", action="store_true", dest="publish", help="use publishing format"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=cmf.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

SL = cmf.setup_logger("script", args.verbose)

full_figsize = cmf.plotting.get_figure_size(args.publish)

figname_base = f"hierarchical_models/PQ"

analysis_params = cmf.utils.read_parameters(args.apf)

HMQ_files = cmf.utils.get_files_in_dir(args.dir)

# TODO set up model selection
if args.model == "simple":
    stan_model_file = "stan/peters_quinlan/pq_simple.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample
        # the likelihood function
        try:
            assert "simple" in args.load_file
        except AssertionError:
            SL.exception(
                f"Using model 'simple', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!",
                exc_info=True,
            )
            raise
        pq_model = cmf.analysis.PQModelSimple.load_fit(
            model_file=stan_model_file,
            fit_files=args.load_file,
            figname_base=figname_base,
        )
    else:
        # sample
        pq_model = cmf.analysis.PQModelSimple(
            model_file=stan_model_file,
            prior_file="stan/peters_quinaln/pq_simple_prior.stan",
            figname_base=figname_base,
        )
else:
    stan_model_file = "stan/peters_quinlan/pq_hierarchy.stan"
    if args.load_file is not None:
        # load a previous sample for improved performance: no need to resample
        # the likelihood function
        try:
            assert "hierarchy" in args.load_file
        except AssertionError:
            SL.exception(
                f"Using model 'hierarchy', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!",
                exc_info=True,
            )
            raise
        pq_model = cmf.analysis.PQModelHierarchy.load_fit(
            model_file=stan_model_file,
            fit_files=args.load_file,
            figname_base=figname_base,
        )
    else:
        # sample
        pq_model = cmf.analysis.PQModelHierarchy(
            model_file=stan_model_file,
            prior_file="stan/peters_quinlan/pq_hierarchy_prior.stan",
            figname_base=figname_base,
        )

pq_model.extract_data(HMQ_files, analysis_params)
pq_model.thin_observations(args.thin)

if args.verbose == "DEBUG":
    pq_model.print_obs_summary()

stan_data = dict(
    N_tot=pq_model.num_obs,
    M1=pq_model.obs["mass1"][0][0],
    M2=pq_model.obs["mass2"][0][0],
)


if args.prior:
    if args.model == "simple":
        idxs = np.argsort(pq_model.obs_collapsed["t"])
        stan_data.update(
            dict(
                t=pq_model.obs_collapsed["t"][idxs],
                a0=pq_model.obs_collapsed["a"][idxs][0],
                e0=pq_model.obs_collapsed["e"][idxs][0],
            )
        )
    else:
        stan_data.update(
            dict(
                N_groups=pq_model.num_groups,
                points_per_group=pq_model.points_per_group,
                t=pq_model.obs_collapsed["t"],
                a0=[a[0] for a in pq_model.obs["a"]],
                e0=[e[0] for e in pq_model.obs["e"]],
            )
        )

    pq_model.sample_prior(
        data=stan_data, sample_kwargs=analysis_params["stan"]["binary_sample_kwargs"]
    )

    fig, ax = plt.subplots(2, 1, sharex="all")
    # ax[1].set_ylim(0,1)
    pq_model.prior_plot(
        xobs="t", yobs="a", xmodel="t", ymodel="a_prior", ax=ax[0], save=False
    )
    pq_model.prior_plot(xobs="t", yobs="e", xmodel="t", ymodel="e_prior", ax=ax[1])

    plt.show()
