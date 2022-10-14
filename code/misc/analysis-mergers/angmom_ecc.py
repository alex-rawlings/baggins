import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import ketjugw
import cm_functions as cmf


kfile = cmf.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/H-H-3.0-0.001/perturbations/002/output")
hmqfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H-H-3.0-0.001/HMQ-cube-H-H-3.0-0.001-002.hdf5"
bins = 50
#tlims = (190.5, 193)
#tlims = (193, 205)

myr = ketjugw.units.yr * 1e6
pc = ketjugw.units.pc

bh1, bh2, merged = cmf.analysis.get_bound_binary(kfile[0])
hmq = cmf.analysis.HMQuantitiesData.load_from_file(hmqfile)

orbit_pars = ketjugw.orbital_parameters(bh1, bh2)
t = orbit_pars["t"]/myr
L = ketjugw.orbital_angular_momentum(bh1, bh2)

sep = cmf.mathematics.radial_separation(bh1.x/pc, bh2.x/pc)
LL = cmf.mathematics.radial_separation(L)
period = 2*np.pi/orbit_pars["n"]/myr

hard_span, hard_idx = cmf.analysis.get_hard_timespan(orbit_pars["t"]/myr, hmq.semimajor_axis, hmq.time_of_snapshot, hmq.semimajor_axis_of_snapshot)

# find the orbit that hardening radius occurs in
hard_time = t[hard_idx]
tlims = (-25+hard_time, 25+hard_time)
mask = np.full_like(t, 0, dtype=bool)
mask[np.logical_and(t>tlims[0], t<tlims[1])] = True

fig, ax = plt.subplots(5,1,sharex="all")
fig2, ax2 = plt.subplots(5,1)

cmf.plotting.binary_param_plot(orbit_pars, ax)
ax2[0].hist(orbit_pars["a_R"][mask]/pc, bins)
ax2[1].hist(orbit_pars["e_t"][mask], bins)

ax[2].semilogy(t[mask], LL[mask])
ax2[2].hist(LL[mask], bins)

ax[3].semilogy(t[mask], sep[mask])
ax2[3].hist(sep[mask], bins)

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
ax[0].set_ylim(130, 210)
ax[1].set_ylim(0.95, 0.97)
ax[2].set_ylim(6.4e11, 7.2e11)
ax[3].set_ylim(5, 300)
ax[4].set_ylim(3, 7)
plt.show()