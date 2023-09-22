import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf
import figure_config


# path to processed data files
e90_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_500K-D_500K-3.720-0.279"
e99_data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_500K-D_500K-3.720-0.028"

# path to raw data
e90_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/500K"
e99_data_raw = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/500K"

# some analysis parameters
analysis_params = cmf.utils.read_parameters("/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml")


# initialise the figure
fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.set_yscale("eccentricity")
thin = 20


# plot the full eccentricity data for some runs
ax.set_xlabel(r"$t'/\mathrm{Myr}$")
ax.set_xscale("log")
ax.set_ylabel(r"$e(t')$")
col = None
start_idx = 2
tmin = 1e-2
t0 = np.full((10,2), np.nan)
kpc = cmf.general.units.kpc

for j, (suite, dp, label) in enumerate(zip(
                (e90_data_raw, e99_data_raw),
                (e90_data_path, e99_data_path),
                (r"$e_0=0.90$", r"$e_0=0.99$")
                )):
    ketju_files = cmf.utils.get_ketjubhs_in_dir(suite)
    HMQ_files = cmf.utils.get_files_in_dir(dp)
    for i, (kf, hf) in enumerate(zip(ketju_files, HMQ_files)):
        print(f"Plotting data from directory {j} run {i}...         ", end="\r")
        bh1, bh2, merged = cmf.analysis.get_bound_binary(kf)
        op = ketjugw.orbital_parameters(bh1, bh2)
        hmq = cmf.analysis.HMQuantitiesBinaryData.load_from_file(hf)
        after_hard_mask = op["a_R"]/kpc < np.nanmedian(hmq.hardening_radius)
        afm0 = after_hard_mask[0]
        t = (op["t"] - op["t"][0]) / cmf.general.units.Myr
        # skip first index as we only plot from t[1:] (note the concatenation)
        t0[i,j] = op["t"][1] / cmf.general.units.Myr
        tmin = min(tmin, t[start_idx])
        t = np.concatenate(([t[0]-t[1]], t)) 
        e = np.concatenate(([1], op["e_t"]))
        after_hard_mask = np.concatenate(([afm0], after_hard_mask))
        if op["e_t"][-1] < 0.6 and np.any(op["e_t"]>0.9):
            print("Adding an artificial point at e=0 for visual appeal...")
            t = np.concatenate((t, np.repeat([2*t[-1]-t[-2]], thin)))
            e = np.concatenate((e, np.repeat([0], thin)))
            after_hard_mask = np.concatenate((after_hard_mask, np.repeat([True], thin)))
        # skip first two points to deal with log scale
        l = ax.plot(t[after_hard_mask][start_idx::thin], e[after_hard_mask][start_idx::thin], c=col, label=(label if i==0 else ""), ls="-")
        col = l[-1].get_color()
        # scatter call so dots are always on top
        ax.scatter(t[after_hard_mask][start_idx], e[after_hard_mask][start_idx], color=col, zorder=l[-1].get_zorder()+0.5, **figure_config.marker_kwargs)
        ax.plot(t[~after_hard_mask][start_idx::thin], e[~after_hard_mask][start_idx::thin], c=col, ls="-", alpha=0.3, zorder=l[-1].get_zorder()-1e-2)
    col = None

ax.legend()
ax.set_xlim(5e-2, 50)
ax.set_ylim(0,1)
print("\nComplete         ")

cmf.plotting.savefig(figure_config.fig_path("eccentricities.pdf"), force_ext=True)
plt.show()
