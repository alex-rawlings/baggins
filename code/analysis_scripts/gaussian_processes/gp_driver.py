import os.path
import numpy as np
import matplotlib.pyplot as plt
import cm_functions as cmf


avg_over_idxs = 50
GP = cmf.analysis.StanModel_2D("stan/gp.stan", "", "gaussian_processes/gp")

theta = []
a_hard = []
e_hard = []

for f in cmf.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/H_500K-H_500K-11.000-0.825"):
    hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
    # protect against instances where no data for bound binary
    try:
        hmq.hardening_radius
    except AttributeError:
        continue
    t = cmf.analysis.first_major_deflection_angle(hmq.prebound_deflection_angles)[0]
    if np.isnan(t): continue
    theta.append(
        [t]
    )
    a_hard.append([np.nanmedian(hmq.hardening_radius)])
    status, hard_idx = hmq.idx_finder(a_hard[-1][0], hmq.semimajor_axis)
    if not status: continue
    idxs = np.r_[hard_idx-avg_over_idxs:hard_idx+avg_over_idxs]
    e_hard.append(
        [np.nanmedian(hmq.eccentricity[idxs])]
    )


GP.obs = dict(
    theta = theta,
    a_hard = a_hard,
    ecc = e_hard
)

GP.transform_obs("theta", "theta_deg", lambda x: x*180/np.pi)

GP.collapse_observations(["theta", "theta_deg", "a_hard", "ecc"])

stan_data = dict(
    theta1 = GP.obs_collapsed["theta"],
    theta_deg = GP.obs_collapsed["theta"] * 180/np.pi,
    ecc = GP.obs_collapsed["ecc"],
    N1 = GP.num_obs,
    N2 = 50
)

stan_data.update(dict(
        theta2 = np.linspace(
            min(stan_data["theta1"]),
            max(stan_data["theta1"]),
            stan_data["N2"]
        )
))
stan_data.update(dict(
    theta2_deg = stan_data["theta2"] * 180/np.pi
))

outdir = os.path.join(cmf.DATADIR, f"stan_files/gps")

GP.sample_model(data=stan_data, sample_kwargs={"output_dir":outdir})



fig, ax = plt.subplots(1,1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\theta\degree$")
ax.set_ylabel(r"$e_\mathrm{h}$")
GP.posterior_plot("theta_deg", "ecc", "theta2_deg", "y", collapsed=True, ax=ax, levels=[25, 68, 95])
ax2 = GP.parameter_corner_plot(["rho", "alpha", "sigma"])
fig2 = ax2.flatten()[0].get_figure()
cmf.plotting.savefig(GP._make_fig_name(GP.figname_base, f"corner_{GP._parameter_corner_plot_counter}"), fig=fig2)

fig3, ax3 = plt.subplots(1,1)
GP.plot_generated_quantity_dist("y", ax=ax3)
ax3.set_xlabel("e")
ax3.set_ylabel("PDF")
fig3 = ax3.get_figure()
cmf.plotting.savefig(GP._make_fig_name(GP.figname_base, f"marginal_eh"), fig=fig3)