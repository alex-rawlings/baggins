import os.path
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import ketjugw
import baggins as bgs

data = bgs.utils.load_data("/users/arawling/projects/collisionless-merger-sample/papers/paper-eccentricity/scripts/data/deflection_angles_e0-0.900.pickle")
#print(data)
#quit()

# units Gyr, 1e10 Msol, kpc
G = 44900
km_per_s = 1.02201216 #kpc/Gyr

def b90(M, v):
    return G*M/(v*km_per_s)**2


def theta_defl(M,v,b):
    return 2 * np.arctan2(b90(M,v),b) * 180/np.pi

def impact(M,v,t):
    return b90(M,v) / np.tan(t*np.pi/180/2)

rng = np.random.default_rng()

# inputs
M = 2e-2
v = 469
b_mean = impact(M,v,60)
print(f"b_mean: {b_mean:.2e}")

bins = 100

fig, ax = plt.subplots(2,1)
ax[0].set_xlabel(r"$b/\mathrm{kpc}$")
ax[1].set_xlabel(r"$\theta/\mathrm{degrees}$")
for axi in ax: axi.set_ylabel("PDF")


for s in (1e-3, 5e-3, 10e-3, 20e-3, 25e-3):
    low_clip = (0-b_mean)/s
    bs = scipy.stats.truncnorm.rvs(low_clip, np.inf, loc=b_mean, scale=s, size=10000, random_state=rng)

    ax[0].hist(bs, bins, histtype="step", density=True, label=f"{s:.1e}")
    ax[1].hist(theta_defl(M,v,bs), bins, histtype="step", density=True)

thetas = np.array(data["thetas"])[np.abs(np.array(data["mass_res"])-10000)<1e-5]
print(thetas)
ax[1].hist(thetas, density=True)

ax[0].legend(title=r"$\sigma_b/\mathrm{kpc}$")
plt.show()