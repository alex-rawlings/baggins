import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ketjugw
import pygad
import cm_functions as cmf



datadirs = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_05",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_07",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_09",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_099",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-095/2M",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-070/2M",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-099/2M",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-097/2M"
]

bound_idxs = 20
myr = cmf.general.units.Myr
kpc = cmf.general.units.kpc
rv_crit = -300
datadir_num = -3

bhfiles = cmf.utils.get_ketjubhs_in_dir(datadirs[datadir_num])

thetas = np.full(len(bhfiles), np.nan)
iqr_b = np.full((2,(len(bhfiles))), np.nan)
median_eccs = np.full(len(bhfiles), np.nan)
iqr_eccs = np.full((2,(len(bhfiles))), np.nan)

# set up a Data frame to store values to
dataframe = pd.DataFrame(index=range(len(bhfiles)), columns=["theta", "e_low", "e_up", "a_hard", "sep", "vx", "vy", "vz"])

for i, bhfile in enumerate(bhfiles):
    #bh1, bh2, _ = cmf.analysis.get_bh_particles(bhfile)
    bh1, bh2 = cmf.analysis.get_binary_before_bound(bhfile)
    bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)

    try:
        bh1_bound, bh2_bound, merged = cmf.analysis.get_bound_binary(bhfile)
    except IndexError:
        print(f"Skipping file {i}")
        continue
    bh1_bound, bh2_bound = cmf.analysis.move_to_centre_of_mass(bh1_bound, bh2_bound)

    _, idxs, sep = cmf.analysis.find_pericentre_time(bh1, bh2, return_sep=True, prominence=0.005)

    theta_d = cmf.analysis.deflection_angle(bh1, bh2, idxs)
    dataframe.loc[i,"theta"], theta_idx = cmf.analysis.first_major_deflection_angle(theta_d, np.pi/4)
    if theta_idx is None:
        print(f"No hard scattering in file {i}, skipping")
        continue
    thetas[i] = dataframe.loc[i,"theta"]
    print(dataframe.loc[i,"theta"])

    op = ketjugw.orbital_parameters(bh1_bound, bh2_bound)
    snapfiles = cmf.utils.get_snapshots_in_dir(os.path.dirname(bhfile))
    snap = pygad.Snapshot(snapfiles[int(len(snapfiles)/2)], physical=True)
    rinfl = max(list(cmf.analysis.influence_radius(snap).values()))
    print(f"Influence radius: {rinfl}")
    ahard = cmf.analysis.hardening_radius(snap.bh["mass"], rinfl)
    print(f"Hardening radius: {ahard} kpc")
    dataframe.loc[i, "a_hard"] = ahard.view(np.ndarray)*1e3
    ahard_idx = cmf.general.get_idx_in_array(ahard, op["a_R"]/kpc)
    print(f"Hardening index: {ahard_idx} / {len(op['a_R'])}")
    _, period_idxs = cmf.analysis.find_idxs_of_n_periods(op["t"][ahard_idx], op["t"], cmf.mathematics.radial_separation(bh1_bound.x, bh2_bound.x), num_periods=11)
    print(f"Period idxs: {period_idxs}")
    m, iqr = cmf.mathematics.quantiles_relative_to_median(op["e_t"][period_idxs[0]:period_idxs[1]])
    median_eccs[i] = m
    iqr_eccs[0,i], iqr_eccs[1,i] = iqr
    dataframe.loc[i,"e_med"] = m
    dataframe.loc[i,"e_low"] = iqr[0][0]
    dataframe.loc[i,"e_up"] = iqr[1][0]
    dataframe.loc[i,"sep"] = sep[idxs[theta_idx]]/ketjugw.units.pc
    dataframe.loc[i, "vx"], dataframe.loc[i, "vy"], dataframe.loc[i, "vz"] = (bh1.v[idxs[theta_idx]] - bh2.v[idxs[theta_idx]])/ketjugw.units.km_per_s

print(dataframe)
dataframe.to_csv(f"dataframe_{datadirs[datadir_num].split('/')[-2]}.csv")

plt.errorbar(thetas*180/np.pi, median_eccs, xerr=None, yerr=iqr_eccs, fmt=".")
plt.xlabel(r"$\theta_\mathrm{defl}$")
plt.ylabel("e")
plt.ylim(0,1)
plt.title(f"{datadirs[datadir_num].split('/')[-2]}")
plt.show()
