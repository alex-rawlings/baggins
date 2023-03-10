import argparse
import yaml
import numpy as np
import matplotlib as plt
from matplotlib import rcParams
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Run Stan model for Quinlan hardening parameter", allow_abbrev=False)
parser.add_argument(type=str, help="path to yml directory file", dest="files")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-m", "--model", help="model to run", type=str, choices=["simple", "hierarchy"], dest="model", default="hierarchy")
parser.add_argument("-s", "--sample", help="sample set", type=str, dest="sample", choices=["mcs", "perturb"], default="mcs")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args = parser.parse_args()


SL = cmf.ScriptLogger("script", console_level=args.verbose)

if args.publish:
    cmf.plotting.set_publishing_style()
    full_figsize = rcParams["figure.figsize"]
    full_figsize[0] *= 2
else:
    full_figsize = None

with open(args.files, "r") as f:
    stan_files = yaml.safe_load(f)

analysis_params = cmf.utils.read_parameters(args.apf)

for family, files in stan_files.items():
    figname_base = f"hierarchical_models/merger_time/{args.sample}/{family}/merger-{family}"

    # load a previous sample for improved performance: no need to resample 
    # the likelihood function
    if args.model == "simple":
        # simple Bayesian regression
        try:
            assert "simple" in files["stan_quinlan"]
            assert "simple" in files["stan_binary"]
        except AssertionError:
            SL.logger.exception(f"Using model 'simple', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        quinlan_model = cmf.analysis.QuinlanModelSimple.load_fit(model_file="stan/quinlan_simple.stan", fit_files=files["stan_quinlan"], figname_base=figname_base)
        kepler_model = cmf.analysis.KeplerModelSimple.load_fit(model_file="stan/binary_simple.stan", fit_files=files["stan_binary"], figname_base=figname_base)
    else:
        # hierarchical model
        try:
            assert "hierarchy" in files["stan_quinlan"]
            assert "hierarchy" in files["stan_binary"]
        except AssertionError:
            SL.logger.exception(f"Using model 'hierarchy', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        quinlan_model = cmf.analysis.QuinlanModelHierarchy.load_fit(model_file="stan/quinlan_hierarchy.stan", fit_files=files["stan_quinlan"], figname_base=figname_base)
        kepler_model = cmf.analysis.KeplerModelHierarchy.load_fit(model_file="stan/binary_hierarchy.stan", fit_files=files["stan_binary"], figname_base=figname_base)

    kepler_model.extract_data(files["HMQ_files"], analysis_params)

    SL.logger.info(f"Number of simulations with usable data (binary): {kepler_model.num_groups}")
    SL.logger.info(f"Number of simulations with usable data (hardening): {quinlan_model.num_groups}")

    stan_data = dict(
        N_groups = len(kepler_model.obs["label"]),
        e_0 = e0
    )

