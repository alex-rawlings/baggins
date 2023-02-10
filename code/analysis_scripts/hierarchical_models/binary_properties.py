import argparse
import os.path
import numpy as np
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
from arviz.labels import MapLabeller
import cm_functions as cmf
from ketjugw.units import unit_length_in_pc, unit_time_in_years


parser = argparse.ArgumentParser(description="Run Stan model for binary properties", allow_abbrev=False)
parser.add_argument(type=str, help="Directory to HMQuantity HDF5 files", dest="dir")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-p", "--prior", help="Plot for prior", action="store_true", dest="prior")
parser.add_argument("-l", "--load", type=str, help="Load previous stan file", dest="load_file", default=None)
parser.add_argument("-P", "--Publish", action="store_true", dest="publish", help="use publishing format")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args = parser.parse_args()

SL = cmf.ScriptLogger("script", console_level=args.verbose)
'''print(f"{1/pc:.3e}")
print(f"{C.parsec:.3e}")
print(f"{C.year:.3e}")
quit()'''

if args.publish:
    cmf.plotting.set_publishing_style()
    full_figsize = rcParams["figure.figsize"]
    full_figsize[0] *= 2
else:
    full_figsize = None

HMQ_files = cmf.utils.get_files_in_dir(args.dir)
with h5py.File(HMQ_files[0], mode="r") as f:
    merger_id = f["/meta"].attrs["merger_id"]
figname_base = f"hierarchical_models/binary/{merger_id}/binary_properties-{merger_id}"

analysis_params = cmf.utils.read_parameters(args.apf)
stan_model_file = "stan/binary_properties.stan"

if args.load_file is not None:
    # load a previous sample for improved performance: no need to resample the
    # likelihood function
    kepler_model = cmf.analysis.StanModel_1D.load_fit(model_file=stan_model_file, fit_files=args.load_file, figname_base=figname_base)
else:
    # sample
    kepler_model = cmf.analysis.StanModel_1D(model_file=stan_model_file, prior_file="stan/binary_prior_2.stan", figname_base=figname_base)

# set up observations
observations = {"angmom":[], "a":[], "e":[], "mass1":[], "mass2":[]}
i = 0
for f in HMQ_files:
    SL.logger.info(f"Loading file: {f}")
    hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
    try:
        idx = hmq.get_idx_in_vec(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
    except ValueError:
        SL.logger.warning(f"No data prior to merger! The requested semimajor axis value is {np.nanmedian(hmq.hardening_radius)}, semimajor_axis attribute is: {hmq.semimajor_axis}. This run will not form part of the analysis.")
        continue
    except AssertionError:
        SL.logger.warning(f"Trying to search for value {analysis_params['bh_binary']['target_semimajor_axis']['value']}, but an AssertionError was thrown. The array bounds are {min(hmq.semimajor_axis)} - {max(hmq.semimajor_axis)}. This run will not form part of the analysis.")
        continue
    t_target = hmq.binary_time[idx]
    target_idx, delta_idxs = cmf.analysis.find_idxs_of_n_periods(t_target, hmq.binary_time, hmq.binary_separation)
    SL.logger.debug(f"For observation {i} found target time between indices {delta_idxs[0]} and {delta_idxs[1]}")
    period_idxs = np.r_[delta_idxs[0]:delta_idxs[1]]
    observations["angmom"].append(hmq.binary_angular_momentum[period_idxs])
    observations["a"].append(hmq.semimajor_axis[period_idxs])
    observations["e"].append(hmq.eccentricity[period_idxs])
    # TODO will need individual bh masses in general
    observations["mass1"].append([hmq.masses_in_galaxy_radius["bh"][0]/2])
    observations["mass2"].append([hmq.masses_in_galaxy_radius["bh"][0]/2])
    i += 1


kepler_model.obs = observations

SL.logger.info(f"Number of simulations with usable data: {kepler_model.num_groups}")
try:
    assert kepler_model.num_groups >= analysis_params["stan"]["min_num_samples"]
except AssertionError:
    SL.logger.exception(f'There are not enough groups to form a valid hierarchical model. Minimum number of groups is {analysis_params["stan"]["min_num_samples"]}, and we have {kepler_model.num_groups}!', exc_info=True)
    raise

# transform observations
G_in_Msun_pc_yr = unit_length_in_pc**3 / unit_time_in_years**2
kepler_model.transform_obs(("mass1", "mass2"), "total_mass", lambda x,y: x+y)
kepler_model.obs["total_mass_long"] = []
offset_factor = np.linspace(1, 1.01, kepler_model.num_groups)
for tm, am in zip(kepler_model.obs["total_mass"], kepler_model.obs["angmom"]):
    kepler_model.obs["total_mass_long"].append(np.repeat(tm, len(am)))
kepler_model.transform_obs("angmom", "angmom_corr", lambda x: x*unit_length_in_pc**2/unit_time_in_years)
kepler_model.transform_obs(("angmom_corr", "total_mass_long"), "angmom_corr_red", lambda l, m: l/np.sqrt(G_in_Msun_pc_yr * m))
kepler_model.transform_obs("angmom_corr_red", "log_angmom_corr_red", lambda x: np.log10(x))
kepler_model.transform_obs("total_mass_long", "log_total_mass_long", lambda x: np.log10(x))

kepler_model.collapse_observations(["log_angmom_corr_red", "a", "e", "total_mass_long", "log_total_mass_long"])

if args.verbose == "DEBUG":
    kepler_model.print_obs_summary()

# initialise the data dictionary
stan_data = {}

if args.prior:
    # create the push-forward distribution for the prior model
    kepler_model.sample_prior(data=stan_data, sample_kwargs=analysis_params["stan"]["sample_kwargs"])
    fig, ax = plt.subplots(1,1, figsize=full_figsize)
    kepler_model.prior_plot(xobs="log_angmom_corr_red", xmodel="log_angmom", ax=ax)
    
    plt.show()
else:
    stan_data.update(dict(
        N_child = kepler_model.obs_len,
        M = kepler_model.obs["mass1"] + kepler_model.obs["mass2"],
        M_reduced = kepler_model.obs["mass1"] * kepler_model.obs["mass2"] / (kepler_model.obs["mass1"] + kepler_model.obs["mass2"])
    ))