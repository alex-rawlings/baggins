import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Run Stan model for Quinlan hardening parameter", allow_abbrev=False)
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument(type=str, help="directory to HMQuantity HDF5 files or csv files", dest="dir")
parser.add_argument(type=str, help="new sample or load previous", choices=["new", "loaded"], dest="type")
parser.add_argument("-p", "--prior", help="plot for prior", action="store_true", dest="prior")
parser.add_argument("-n", "--num", type=int, help="number of predictive sample points", dest="num_predpoints", default=50)
parser.add_argument("-d", "--dir", type=str, action="append", default=[], dest="extra_dirs", help="other directories to compare")
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbosity", default="INFO", help="verbosity level")
args = parser.parse_args()


SL = cmf.ScriptLogger("script", args.verbosity)

if args.type == "new":
    hmq_dirs = []
    hmq_dirs.append(args.dir)
    if args.extra_dirs:
        hmq_dirs.extend(args.extra_dirs)
        SL.logger.debug(f"Directories are: {hmq_dirs}")
else:
    hmq_dirs = None
analysis_params = cmf.utils.read_parameters(args.apf)

figname_base = "gaussian_processes"
model_file = "stan/gp.stan"

if args.type == "loaded":
    GP = cmf.analysis.DeflectionAngleGP.load_fit(model_file, args.dir, figname_base)
else:
    GP = cmf.analysis.DeflectionAngleGP(model_file, "", figname_base)

GP.extract_data(analysis_params, hmq_dirs)
e_ini = f"e0-{GP.e_ini:.2f}".replace(".", "")
GP.figname_base = os.path.join(GP.figname_base, f"{e_ini}/{e_ini}")

GP.set_stan_dict(args.num_predpoints)

outdir = os.path.join(cmf.DATADIR, f"stan_files/gps/{e_ini}")

GP.sample_model(sample_kwargs={"output_dir":outdir})
GP.all_plots()