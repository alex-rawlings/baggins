import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs




fig2, ax2 = plt.subplots(1,1)

ax2.set_xlabel(r"$t/\mathrm{Myr}$")
ax2.set_ylabel(r"$\sigma_\mathrm{pos}/\mathrm{kpc}$")


only_scatter_angle = True


if False:
    data = {}
    fig, ax = plt.subplots(2,1,sharex="all")
    mainpath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090"
    mainpath.rstrip("/")
    print(f"Analysing: {mainpath}")
    dirs = [
            os.path.join(mainpath, "250K"),
            os.path.join(mainpath, "500K"),
            os.path.join(mainpath, "1M"),
            os.path.join(mainpath, "2M"),
            os.path.join(mainpath, "4M"),
    ]
    threshold_angle = 45 if "90" in os.path.basename(mainpath)[2:] else 90
    threshold_angle *= np.pi/180
    for d in dirs:
        seps = []
        hard_scatter_time = []
        ketjufiles = bgs.utils.get_ketjubhs_in_dir(d)
        for i, kf1 in enumerate(ketjufiles):
            bhA1, bhA2 = bgs.analysis.get_binary_before_bound(kf1)
            bhA1, bhA2 = bgs.analysis.move_to_centre_of_mass(bhA1, bhA2)
            
            if only_scatter_angle:
                peri_times, peri_idx, sep = bgs.analysis.find_pericentre_time(bhA1, bhA2, return_sep=True, prominence=0.005)
                hard_angles = bgs.analysis.deflection_angle(bhA1, bhA2, peri_idx)
                print(hard_angles)
                idx = bgs.analysis.first_major_deflection_angle(hard_angles, threshold_angle)[1]
                if idx is not None:
                    hard_scatter_time.append(peri_times[idx]/bgs.general.units.Myr)
                continue

            for j, kf2 in enumerate(ketjufiles[i+1:], start=i+1):
                print(f"Determining offset between {i} and {j}")
                bhB1, bhB2 = bgs.analysis.get_binary_before_bound(kf2)
                bhB1, bhB2 = bgs.analysis.move_to_centre_of_mass(bhB1, bhB2)

                # truncate to common length
                bhA1 = bhA1[:len(bhB1)]
                bhA2 = bhA2[:len(bhB1)]
                bhB1 = bhB1[:len(bhA1)]
                bhB2 = bhB2[:len(bhA1)]

                for k, (axi, bh1, bh2) in enumerate(zip(ax, (bhA1, bhA2), (bhB1, bhB2))):
                    sep = bgs.mathematics.radial_separation(bh1.x-bh1.x[0,:], bh2.x-bh2.x[0,:]) / bgs.general.units.kpc
                    t = bh1.t/bgs.general.units.Myr
                    #axi.plot(t, sep)
                    if k==0:
                        seps.append(sep)
                        break
        
        if only_scatter_angle:
            print(f"Avg scatter time: {np.nanmean(hard_scatter_time):.2f}")
        else:
            seps = np.array(seps)
            data[os.path.basename(d)] = dict(
                t = t,
                seps = seps
            )
    if not only_scatter_angle:
        bgs.utils.save_data(data, f"sigma-BH-pos-e{os.path.basename(mainpath)[3:]}.pickle")
        plt.show()
else:
    cols = bgs.plotting.mplColours()

    for ecc, ls in zip(("90", "99"), ("-", "--")):
        data = bgs.utils.load_data(f"sigma-BH-pos-e{ecc}.pickle")

        all_min_len = np.inf
        for k, v in data.items():
            min_len = np.inf
            for a in v["seps"]:
                if len(a) < min_len: min_len = len(a)
            for i in range(len(v["seps"])):
                v["seps"][i] = v["seps"][i][:min_len]
            v["t"] = v["t"][:min_len]
            data[k]["seps"] = np.vstack(data[k]["seps"])
            if data[k]["seps"].shape[1] < all_min_len:
                all_min_len = data[k]["seps"].shape[1]

        for i, (k, v) in enumerate(data.items()):
            ax2.semilogy(v["t"], np.nanstd(v["seps"], axis=0), label=(k if ecc=="90" else ""), ls=ls, c=cols[i])
        ax2.set_ylim(1e-4, ax2.get_ylim()[1])
    ax2.axvline(23.5, label="Avg e99 scatter time", c="k", ls="--")
    ax2.axvline(33, label="Avg e90 scatter time", c="k", ls="-")
    ax2.legend()
    plt.show()