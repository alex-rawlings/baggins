import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import baggins as bgs
import pygad


file_low = "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output/ketju_bhs.hdf5"
file_highN = "/scratch/pjohanss/arawling/collisionless_merger/regen-test/noncentred/high_softening/output/ketju_bhs.hdf5"
file_high = "/scratch/pjohanss/arawling/collisionless_merger/regen-test/recentred/high-softening/output/ketju_bhs_cp.hdf5"
extraction_snap = "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output/A05-C05-3.0-0.001_014.hdf5"

snap = pygad.Snapshot(extraction_snap)
snap.to_physical_units()
snaptime = bgs.general.convert_gadget_time(snap)

kpc = ketjugw.units.pc * 1e3
myr = ketjugw.units.yr * 1e6
kmps = ketjugw.units.km_per_s

cols = bgs.plotting.mplColours()
lstyles = bgs.plotting.mplLines()
axlabels = ["x", "y", "z"]
alpha = 0.3

fig, ax = plt.subplots(3,2, sharex="all", figsize=(6,4))
for ind, (this_file, l) in enumerate(zip((file_low, file_highN, file_high), lstyles)):
    bhs = ketjugw.data_input.load_hdf5(this_file)
    for ind2, bh in enumerate(bhs.values()):
        if ind == 0 and ind2 == 0:
            time_offset = 0
            last_idx = np.argmin(np.abs(bh.t/(1e3*myr) - snaptime))
            print("last idx: {}".format(last_idx))
        for i in range(3):
            if ind2 ==0 and i==0:
                if ind==0:
                    labval="Low"
                elif ind == 1:
                    labval = "High (NC)"
                else:
                    labval="High (RC)"
            else:
                labval=""
            ax[i, 0].set_ylabel(r"{}/kpc".format(axlabels[i]))
            ax[i, 1].set_ylabel(r"v$_{}$/km/s".format(axlabels[i]))
            ax[i, 0].plot(bh.t[:last_idx]/myr + time_offset, bh.x[:last_idx,i]/kpc, markevery=[-1], marker="o", c=cols[ind2], ls=l, label=labval)
            ax[i, 0].plot(bh.t[last_idx:]/myr + time_offset, bh.x[last_idx:,i]/kpc, c=cols[ind2], ls=l, alpha=alpha)
            ax[i, 1].plot(bh.t[:last_idx]/myr +time_offset, bh.v[:last_idx,i]/kmps, markevery=[-1], marker="o", c=cols[ind2], ls=l)
            ax[i, 1].plot(bh.t[last_idx:]/myr +time_offset, bh.v[last_idx:,i]/kmps, c=cols[ind2], ls=l, alpha=alpha)
        if ind==0 and ind2 == 1:
            time_offset = bh.t[last_idx]/myr
            last_idx = -1
ax[0,0].legend(loc="upper left")
for axi in ax[-1, :]:
    axi.set_xlabel("t/Myr")

plt.show()
