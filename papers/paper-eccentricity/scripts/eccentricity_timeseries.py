import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ketjugw
import cm_functions as cmf
import figure_config


# path to processed data files
e90_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_500K-D_500K-3.720-0.279"
e99_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_4M-D_4M-3.720-0.028"

# path to raw data
e90_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/500K"
e99_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/4M"

# some analysis parameters
analysis_params = cmf.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml")


# initialise the figure
fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.set_yscale("eccentricity")


# plot the full eccentricity data for some runs
ax.set_xlabel(r"$t'/\mathrm{Myr}$")
ax.set_ylabel(r"$e$")
col = None
start_idx = 2
tmin = np.inf
for suite, label in zip((e90_data_raw, e99_data_raw), (r"$e_0=0.90$", r"$e_0=0.99$")):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(suite)
    for i, kf in enumerate(ketju_files):
        bh1, bh2, merged = cmf.analysis.get_bound_binary(kf)
        op = ketjugw.orbital_parameters(bh1, bh2)
        t = (op["t"] - op["t"][0]) / cmf.general.units.Myr
        tmin = min(tmin, t[start_idx])
        t = np.concatenate(([t[0]-t[1]], t)) 
        e = np.concatenate(([1], op["e_t"]))
        # skip first two points to deal with log scale
        l = ax.plot(t[start_idx:], e[start_idx:], zorder=10, c=col, label=(label if i==0 else ""), ls="-")
        col = l[-1].get_color()
    col = None
ax.set_xlim(tmin, 50)
ax.legend()
#ax.set_xscale("symlog", linthresh=2)
ax.set_ylim(0,1)



for i, (data_path, col) in enumerate(zip((e90_data_path, e99_data_path), cmf.plotting.mplColours())):
    # extract data
    HMQ_files = cmf.utils.get_files_in_dir(data_path)
    km = cmf.analysis.KeplerModelHierarchy("", "", "")
    km.extract_data(HMQ_files, analysis_params)
    mean_t_h = np.mean([np.mean(t) for t in km.obs["t"]])
    ax.annotate("", (mean_t_h, 1-5e-4), (mean_t_h, 1), arrowprops={"arrowstyle":"-|>", "fc":col, "ec":col})

cmf.plotting.savefig(figure_config.fig_path("eccentricities.pdf"), force_ext=True)