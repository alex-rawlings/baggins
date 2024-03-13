import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pygad
import baggins as bgs


snapfiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output/A05-C05-3.0-0.001_014.hdf5", 
    "/scratch/pjohanss/arawling/collisionless_merger/regen-test/recentred/high-softening/ACH.hdf5"
    #"/scratch/pjohanss/arawling/collisionless_merger/regen-test/recentred/high-softening/output/ACH-HS_011.hdf5"
    ]
dm_soft = [0.100, 0.100] #kpc
bin_width = 100 #kpc
labval = ["low-res-", "high-res-"]

fig, ax = plt.subplots(3,1, sharex="all")
for ind, snapfile in enumerate(snapfiles):
    snap = pygad.Snapshot(snapfile, physical=True)
    star_id_masks = bgs.analysis.get_all_id_masks(snap)
    dm_id_masks = bgs.analysis.get_all_id_masks(snap, family="dm")
    xcoms= bgs.analysis.get_com_of_each_galaxy(snap, initial_radius=100, masks=dm_id_masks, family="dm")
    vcoms = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcoms, masks=dm_id_masks, family="dm")
    Rvirs, Mvirs = bgs.analysis.get_virial_info_of_each_galaxy(snap, xcoms, masks=[star_id_masks, dm_id_masks])
    for bhid in dm_id_masks.keys():
        xcom = xcoms[bhid]
        vcom = vcoms[bhid]
        id_mask = dm_id_masks[bhid]
        Rvir = Rvirs[bhid]
        snap["pos"] -= xcom
        snap["vel"] -= vcom
        subsnap = snap[id_mask]
        bins = np.arange(0, np.max(subsnap["r"]), bin_width)
        vel_magnitudes = bgs.mathematics.radial_separation(subsnap["vel"])
        particles_per_bin,*_ = np.histogram(subsnap["r"], bins=bins)
        mean_vel_per_bin, bin_edges, *_ = scipy.stats.binned_statistic(subsnap["r"], vel_magnitudes, statistic="mean", bins=bins)
        sd_vel_per_bin, bin_edges, *_ = scipy.stats.binned_statistic(subsnap["r"], vel_magnitudes, statistic="std", bins=bins)
        R_of_bins = bgs.mathematics.get_histogram_bin_centres(bin_edges)

        mask = particles_per_bin > 10
        particles_per_bin = particles_per_bin[mask]
        R_of_bins = R_of_bins[mask]
        mean_vel_per_bin = mean_vel_per_bin[mask]
        sd_vel_per_bin = sd_vel_per_bin[mask]

        delta_V2 = 8*mean_vel_per_bin**2/particles_per_bin * np.log(R_of_bins/dm_soft[ind])
        sd_delta_V2 = 8*sd_vel_per_bin**2/particles_per_bin * np.log(R_of_bins/dm_soft[ind])

        n_relax = particles_per_bin / (8 * np.log(R_of_bins/dm_soft[ind]))
        t_relax = n_relax * R_of_bins / mean_vel_per_bin *0.976 #Gyr
        sd_t_relax_L = n_relax * R_of_bins / (mean_vel_per_bin+sd_vel_per_bin) * 0.976 #Gyr
        sd_t_relax_U = n_relax * R_of_bins / (mean_vel_per_bin-sd_vel_per_bin) * 0.976 #Gyr
        
        l,*_=ax[0].loglog(R_of_bins, delta_V2, label=labval[ind]+str(bhid))
        ax[0].fill_between(R_of_bins, y1=delta_V2+sd_delta_V2, y2=delta_V2-sd_delta_V2, alpha=0.3)
        ax[0].axvline(Rvir, c=l.get_color())
        ax[1].loglog(R_of_bins, t_relax)
        ax[1].fill_between(R_of_bins, sd_t_relax_L, sd_t_relax_U, alpha=0.3)
        ax[2].loglog(R_of_bins, particles_per_bin)

        #revert back to original coordinates
        snap["pos"] += xcom
        snap["vel"] += vcom

ax[0].text(0.1, 0.5, r"$\Delta v^2(r) \simeq \frac{8v(r)^2}{N(r)}\ln\left(\frac{r}{\varepsilon_\mathrm{DM}}\right)$", transform=ax[0].transAxes)
ax[0].legend()
ax[-1].set_xlabel("Radius/kpc")
ax[0].set_ylabel(r"$\Delta v^2(r)$")
ax[1].set_ylabel(r"t$_\mathrm{relax}$/Gyr")
ax[2].set_ylabel("Particle Count")
plt.show()