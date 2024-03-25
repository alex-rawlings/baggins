import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


#plot, for both dm and stars, radius containing x particles against eta

main_path = "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001"
data_paths = [
    "perturbations_eta_0020",
    "perturbations_eta_0005",
    "perturbations_eta_0002"
]

particle_numbers = [100, 200, 400, 800, 1600]

cols = bgs.plotting.mplColours()
fig, ax = plt.subplots(1,2, sharex="row", sharey="row")

for i, data_path in enumerate(data_paths):
    eta_path = os.path.join(main_path, data_path)
    eta = eta_path.split("_")[-1]
    eta = float("{}.{}".format(eta[0], eta[1:]))
    print("Determining for eta={}".format(eta))
    particle_number_dict = dict(
        stars = {key: np.full(10, np.nan) for key in particle_numbers},
        dm = {key: np.full(10, np.nan) for key in particle_numbers}
    )
    
    for j in range(10):
        perturbation = "{:03d}".format(j)
        snaplist = bgs.utils.get_snapshots_in_dir(os.path.join(eta_path, "{}/output/".format(perturbation)))
        snap = pygad.Snapshot(snaplist[-1], physical=True)
        merged, remnant_id = bgs.analysis.determine_if_merged(snap)
        initial_com_guess = pygad.analysis.center_of_mass(snap.stars)
        """if merged:
            initial_com_guess = snap.bh["pos"][snap.bh["ID"]==remnant_id]
        else:
            initial_com_guess = None"""
        #ensure CoMs are sufficiently close that the system is like one system
        star_id_masks = bgs.analysis.get_all_id_masks(snap)
        xcoms = bgs.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks, initial_guess=initial_com_guess)
        #print(xcoms)
        com_separation = bgs.mathematics.radial_separation(*list(xcoms.values()))[0]
        if com_separation > 1:
            print("CoM separation > 1 kpc --> skipping")
            continue
        for k, family in enumerate(("stars", "dm")):
            subsnap = getattr(snap, family)
            #let's just take the mean com for now, should be pretty close
            xcom = np.nanmean(np.array(list(xcoms.values())), axis=0)
            all_separations = bgs.mathematics.radial_separation(subsnap["pos"], xcom)
            all_separations.sort()
            for n in particle_numbers:
                particle_number_dict[family][n][j] = all_separations[n]
    #print(particle_number_dict)
    for l, key in enumerate(particle_number_dict["stars"].keys()):
        for k, family in enumerate(particle_number_dict.keys()):
            xval = np.log10(eta)
            yval = np.log10(particle_number_dict[family][key])
            ax[k].errorbar(xval, np.nanmean(yval), yerr=np.nanstd(yval),
                            fmt="o", markerfacecolor=cols[l], 
                            markeredgecolor=cols[l], ecolor=cols[l],
                            label=(key if  i==0 else ""))
ax[0].set_xlabel(r"$\log(\eta)$")
ax[1].set_xlabel(r"$\log(\eta)$")
ax[0].set_title("Stars")
ax[1].set_title("DM")
ax[0].set_ylabel("log(r/kpc)")
ax[0].legend()
plt.show()
quit()
