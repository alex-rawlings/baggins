import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import ketjugw

# code hacked from HMQuantitiesBinary

data_dirs = [
    "/scratch/pjohanss/attekeit/df_test/galaxy_mergers/ketju/output_gasless_merger_10000_run_",
    "/scratch/pjohanss/attekeit/df_test/galaxy_mergers/ketju/output_gasless_merger_1000_run_",
    "/scratch/pjohanss/attekeit/df_test/galaxy_mergers/ketju/output_gasless_merger_100_run_",
]

fig1, ax1 = plt.subplots(1,1)
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$e$")

fig2, ax2 = plt.subplots(1,1)
ax2.set_xlabel(r"$t$/Myr")
ax2.set_ylabel(r"$e$")
cols = bgs.plotting.mplColours()

for i, d in enumerate(data_dirs):
    #if i<2: continue
    print(f"Reading from base directory: {d}")
    thetas = np.full(5, np.nan)
    eccs = np.full_like(thetas, np.nan)

    for run_num in range(5):
        print(f"  {run_num+1}")
        data_dir = f"{d}{run_num+1}"
        ketju_file = bgs.utils.get_ketjubhs_in_dir(data_dir)[0]

        # get the BH particles
        bh1, bh2, merged = bgs.analysis.get_bound_binary(ketju_file)
        orbit_pars = ketjugw.orbital_parameters(bh1, bh2)
        bh1_pb, bh2_pb, bound_state = bgs.analysis.get_binary_before_bound(ketju_file)
        if bound_state == "oscillate":
            print("Binary is not reliably bound, skipping")
            continue

        # pericentre deflection angle before binary is bound
        bh1_pb, bh2_pb = bgs.analysis.move_to_centre_of_mass(bh1_pb, bh2_pb)
        try:
            _, peri_idxs = bgs.analysis.find_pericentre_time(bh1_pb, bh2_pb, prominence=0.005)
            prebound_deflection_angles = bgs.analysis.deflection_angle(bh1_pb, bh2_pb, peri_idxs)
        except:
            print(f"Unable to determine pericentre times before binary is bound!")
            prebound_deflection_angles = []
        ecc_idx = int(len(orbit_pars["e_t"])/5)
        ecc = orbit_pars["e_t"][:ecc_idx]
        thetas[run_num] = bgs.analysis.first_major_deflection_angle(prebound_deflection_angles, np.pi/6)[0]
        eccs[run_num] = np.nanmedian(ecc)
        ax2.plot(orbit_pars["t"][:ecc_idx]/bgs.general.units.Myr, ecc, c=cols[i])
    ax1.scatter(np.degrees(thetas), eccs, label=os.path.basename(d).split("_")[-3])
ax1.legend()
for axi in ax1, ax2:
    axi.set_ylim(0,1)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "deflection_angles/atte_minor_mergers_theta"), fig=fig1)
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "deflection_angles/atte_minor_mergers_ecc"), fig=fig2)
plt.show()
