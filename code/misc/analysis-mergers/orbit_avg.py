import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import ketjugw

import baggins as bgs

ketju_file = bgs.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/H-H-3.0-0.001/perturbations/002")[0]

myr = ketjugw.units.yr * 1e6
pc = ketjugw.units.pc

bh1, bh2, merged = bgs.analysis.get_bound_binary(ketju_file=ketju_file)

orbit_pars = ketjugw.orbital_parameters(bh1, bh2)
t = orbit_pars["t"]/myr
L = bgs.mathematics.radial_separation(ketjugw.orbital_angular_momentum(bh1, bh2))
sep = bgs.mathematics.radial_separation(bh1.x/pc, bh2.x/pc)



def find_period_idxs(tval, tarr, sep, num_orbits=1):
    idx = bgs.general.get_idx_in_array(tval, tarr)
    y = np.diff(np.sign(np.diff(sep)))
    found_peaks = False
    multiplier = 1
    max_idx = len(sep)-1
    while not found_peaks:
        idxs = np.r_[max(0, idx-10*multiplier):min(max_idx, idx+10*multiplier)]
        peaks = np.where(y[idxs]==2)[0]
        if len(peaks) > 2*num_orbits: 
            found_peaks = True
            peaks_rel = idxs[0]+peaks - idx
            start_idx = peaks_rel[np.where(peaks_rel<0, peaks_rel, -np.inf).argmax()-num_orbits//2] + idx
            end_idx = peaks_rel[np.where(peaks_rel>=0, peaks_rel, np.inf).argmin()+num_orbits//2] + idx
        else:
            multiplier *= 2
    return idx, y, start_idx, end_idx


idx, y, start_idx, end_idx = find_period_idxs(300, t, sep, 4)
print(idx, start_idx, end_idx)

fig, ax = plt.subplots(2,1, sharex="all")

ax[0].set_xlim(290, 306)
ax[0].set_ylim(0, 120)
#ax[1].set_ylim(4.5e11, 5.5e11)

for axi in ax:
    axi.axvline(300, c="k")
ax[0].plot(t, sep)
ax[0].plot(t[start_idx:end_idx], sep[start_idx:end_idx])
ax[0].scatter(t[idx], sep[idx], zorder=10)
ax[1].plot(t[2:], y)
ax[1].scatter(t[idx], y[idx], zorder=10)
ax[1].plot(t[start_idx:end_idx], y[start_idx:end_idx])

plt.show()