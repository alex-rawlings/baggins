import numpy as np
import matplotlib.pyplot as plt
import cmdstanpy
from scipy.spatial import distance
import baggins as bgs


true_mag = np.loadtxt("mag_data.txt")
grid_x, grid_y = np.meshgrid(np.arange(true_mag.shape[0]), np.arange(true_mag.shape[1]))
pixel_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
# Define a distance threshold for local neighborhoods
d_max = 2.0

# Compute Euclidean distance from the center
center = np.array([true_mag.shape[0] / 2, true_mag.shape[1] / 2])
distances = np.linalg.norm(pixel_coords - center, axis=1)

idxs = np.r_[0:len(distances):75]
print(idxs)
N = len(distances[idxs])
print(N)

data = {
    "N1": N,
    "x1": distances[idxs],
    "y1": true_mag.flatten()[idxs],  # observed pixel counts
    "N2": 100*N,
    "x2": np.linspace(min(distances), max(distances), 100*N)
}

# Save the Stan model as "gp_pixel_model.stan" before this
model = cmdstanpy.CmdStanModel(stan_file="/users/arawling/projects/collisionless-merger-sample/baggins/stan/gaussian-process/gp_analytic.stan")
fit = model.sample(data=data, chains=4, iter_sampling=2000)

# Extract posterior samples for the latent GP function f
posterior_f = fit.stan_variable('f')
f_mean = posterior_f.mean(axis=0)

local_outliers = []
for idx in range(len(pixel_coords)):
    # Find neighboring pixels within d_max distance
    dists = distance.cdist([pixel_coords[idx]], pixel_coords)[0]
    neighbors = np.where(dists < d_max)[0]

    # Compute posterior statistics for the neighborhood
    local_f_mean = posterior_f[:, neighbors].mean(axis=1)
    local_f_lower = np.percentile(local_f_mean, 2.5)
    local_f_upper = np.percentile(local_f_mean, 97.5)

    # Check if the observed value is outside the local credible interval
    if true_mag[idx] < local_f_lower or true_mag[idx] > local_f_upper:
        local_outliers.append(idx)

print("Local outlier pixel indices:", local_outliers)
print("Local outlier pixel coordinates:", pixel_coords[local_outliers])

plt.imshow(true_mag, cmap='viridis')
plt.colorbar(label='Pixel Counts')
for idx in local_outliers:
    x, y = pixel_coords[idx]
    plt.plot(y, x, 'ro', markersize=8, label='Local Outlier' if idx == local_outliers[0] else "")
plt.title("Local Outliers in Pixel Grid")
plt.legend()
plt.show()

