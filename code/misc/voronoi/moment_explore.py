import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import baggins as bgs


fig, ax = plt.subplots(2,2, sharex="all")

x = np.linspace(-5, 5, 500)
h3_h4_max = 0.2
Vmax = 2
sigma_max = 5
N = 5

# mean velocity
Vs = np.linspace(-Vmax, Vmax, N)
cmapper, sm = bgs.plotting.create_normed_colours(Vs[0], Vs[-1], cmap="icefire")
for _V in tqdm(Vs):
    y = bgs.analysis.gauss_hermite_function(x, _V, 1, 0, 0)
    ax[0,0].plot(x, y, c=cmapper(_V), lw=2)
plt.colorbar(sm, ax=ax[0,0], label=r"$V$")

# sigma
sigmas = np.linspace(1, sigma_max, N)
cmapper, sm = bgs.plotting.create_normed_colours(sigmas[0], sigmas[-1], cmap="flare_r")
for _s in tqdm(sigmas):
    y = bgs.analysis.gauss_hermite_function(x, 0, _s, 0, 0)
    ax[0,1].plot(x, y, c=cmapper(_s), lw=2)
plt.colorbar(sm, ax=ax[0,1], label=r"$\sigma$")

# h3
h3s = np.linspace(-h3_h4_max, h3_h4_max, N)
cmapper, sm = bgs.plotting.create_normed_colours(h3s[0], h3s[-1], cmap="icefire")
for _h3 in tqdm(h3s):
    y = bgs.analysis.gauss_hermite_function(x, 0, 1, _h3, 0)
    ax[1,0].plot(x, y, c=cmapper(_h3), lw=2)
plt.colorbar(sm, ax=ax[1,0], label=r"$h_3$")

# h4
h4s = np.linspace(-h3_h4_max, h3_h4_max, N)
cmapper, sm = bgs.plotting.create_normed_colours(h4s[0], h4s[-1], cmap="icefire")
for _h4 in tqdm(h4s):
    y = bgs.analysis.gauss_hermite_function(x, 0, 1, 0, _h4)
    ax[1,1].plot(x, y, c=cmapper(_h4), lw=2)
plt.colorbar(sm, ax=ax[1,1], label=r"$h_4$")

for axi in ax[1,:]:
    axi.set_xlabel("LOS velocity")
for axi in ax[:,0]:
    axi.set_ylabel("PDF")

bgs.plotting.savefig("moment_vary.pdf", force_ext=True)
plt.show()