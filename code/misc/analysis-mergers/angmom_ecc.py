import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf


kfile = cmf.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/002/output")
hmqfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/A-C-3.0-0.05/HMQ-cube-A-C-3.0-0.05-002.hdf5"
bins = 50
#tlims = (190.5, 193)
tlims = (180, 200)

myr = ketjugw.units.yr * 1e6
pc = ketjugw.units.pc

bh1, bh2, merged = cmf.analysis.get_bound_binary(kfile[0])
hmq = cmf.analysis.HMQuantitiesData.load_from_file(hmqfile)

orbit_pars = ketjugw.orbital_parameters(bh1, bh2)
L = ketjugw.orbital_angular_momentum(bh1, bh2)

hard_span, hard_idx = cmf.analysis.get_hard_timespan(orbit_pars["t"]/myr, hmq.semimajor_axis, hmq.time_of_snapshot, hmq.semimajor_axis_of_snapshot)

t = orbit_pars["t"]/myr
mask = np.full_like(t, 0, dtype=bool)
mask[np.logical_and(t>tlims[0], t<tlims[1])] = True

fig, ax = plt.subplots(5,1,sharex="all")
fig2, ax2 = plt.subplots(5,1)

cmf.plotting.binary_param_plot(orbit_pars, ax)
ax2[0].hist(orbit_pars["a_R"][mask]/pc, bins)
ax2[1].hist(orbit_pars["e_t"][mask], bins)
LL = cmf.mathematics.radial_separation(L)
ax[2].semilogy(t[mask], LL[mask])
ax2[2].hist(LL[mask], bins)
sep = cmf.mathematics.radial_separation(bh1.x/pc, bh2.x/pc)
ax[3].semilogy(t[mask], sep[mask])
ax2[3].hist(sep[mask], bins)
period = 2*np.pi/orbit_pars["n"]/myr
ax[4].plot(t[mask], period[mask])
ax2[4].hist(period[mask], bins)
for axi in ax:
    axi.axvline(t[hard_idx], c="k")

ax[1].set_xlabel("")
for i, label in enumerate(("a/pc", "e", "L", "sep/pc", "Period/Myr")):
    ax[i].set_ylabel(label)
    ax2[i].set_xlabel(label)
ax[4].set_xlabel("t/Myr")

ax[0].set_xlim(*tlims)
ax[0].set_ylim(58, 64)
ax[1].set_ylim(0.845, 0.865)
ax[2].set_ylim(1.42e12, 1.47e12)
ax[3].set_ylim(5, 200)
ax[4].set_ylim(0.52, 0.61)
plt.show()