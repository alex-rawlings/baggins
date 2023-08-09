# Gaussian process example following
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import cm_functions as cmf

avg_over_idxs = 50

# load the data
GP = cmf.analysis.StanModel_2D("stan/gp.stan", "", "gaussian_processes/gp")

theta = []
e_hard = []

for f in cmf.utils.get_files_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/H_500K-H_500K-11.000-0.825"):
    hmq = cmf.analysis.HMQuantitiesData.load_from_file(f)
    try:
        hmq.hardening_radius
    except AttributeError:
        continue
    t = cmf.analysis.first_major_deflection_angle(hmq.prebound_deflection_angles)[0]
    if np.isnan(t): continue
    theta.append(
        [t]
    )
    status, hard_idx = hmq.idx_finder(np.nanmedian(hmq.hardening_radius), hmq.semimajor_axis)
    if not status: continue
    idxs = np.r_[hard_idx-avg_over_idxs:hard_idx+avg_over_idxs]
    e_hard.append(
        [np.nanmedian(hmq.eccentricity[idxs])]
    )


GP.obs = dict(
    theta = theta,
    ecc = e_hard
)

GP.transform_obs("theta", "theta_deg", lambda x: x*180/np.pi)

GP.collapse_observations(["theta", "theta_deg", "ecc"])

# fit the GP
x = np.array(GP.obs_collapsed["theta"])
idx = np.argsort(x)
x = x[idx].reshape(-1,1)
y = GP.obs_collapsed["ecc"]
y = y[idx]
kernel = 1 * kernels.RBF(length_scale=0.5, length_scale_bounds=(1e-2, 5))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(x,y)

# make mean predictions
xp = np.linspace(min(x), max(x), 1000)
mean_pred, std_pred = gp.predict(xp, return_std=True)

plt.scatter(x,y)
plt.plot(xp, mean_pred)
fc = None
for fac, a in zip((3, 1), (0.3, 0.6)):
    f = plt.fill_between(
        xp.ravel(),
        mean_pred - fac * std_pred,
        mean_pred + fac * std_pred,
        alpha=a, 
        fc = fc
    )
    fc = f.get_facecolor()

plt.show()