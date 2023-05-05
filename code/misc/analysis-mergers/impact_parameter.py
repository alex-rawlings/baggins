import os.path
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
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
myr = 1e6 * ketjugw.units.yr
rv_crit = -200
datadir_num = -2

bhfiles = cmf.utils.get_ketjubhs_in_dir(datadirs[datadir_num])

median_b = np.full(len(bhfiles), np.nan)
iqr_b = np.full((2,(len(bhfiles))), np.nan)
median_eccs = np.full(len(bhfiles), np.nan)
iqr_eccs = np.full((2,(len(bhfiles))), np.nan)

for i, bhfile in enumerate(bhfiles):
    #bh1, bh2, _ = cmf.analysis.get_bh_particles(bhfile)
    bh1, bh2 = cmf.analysis.get_binary_before_bound(bhfile)
    bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)

    '''energy = ketjugw.orbital_energy(bh1, bh2)
    mask = energy>0
    plt.plot(mask)
    plt.show()'''

    b, theta = cmf.analysis.impact_parameter(bh1, bh2)
    bmag = cmf.mathematics.radial_separation(b)
    #y = np.diff(np.sign(np.diff(bmag[mask], prepend=0)), prepend=0)
    #idxs = np.where(y==-2)[0][-1]-1
    #idxs = cmf.general.get_idx_in_array(23.9, bh1.t[mask]/myr)

    _, idxs, sep = cmf.analysis.find_pericentre_time(bh1, bh2, return_sep=True, prominence=0.005)
    print(idxs)
    #idxs = idxs[:2]

    apo_idxs = []
    for _idx1, _idx2 in zip(idxs[:-1], idxs[1:]):
        apo_idxs.append(
            np.argmax(sep[_idx1:_idx2]) + _idx1
        )

    #apo_idx = np.argmax(sep[np.r_[idxs[-2]:idxs[-1]]]) + idxs[-2]
    #idxs = np.concatenate((idxs, [apo_idx]))
    #print(apo_idx)

    '''half_apo_peri_idx = cmf.general.get_idx_in_array(
        (sep[apo_idx] + sep[idxs[1]])/2,
        sep[np.r_[apo_idx:idxs[1]]]
    ) + apo_idx'''

    '''for q in (0.5, 0.25, 0.10, 0.05, 0.01):
        _idx = cmf.general.get_idx_in_array(
            np.nanquantile(np.log10(bmag[apo_idx:idxs[1]]), q),
            np.log10(bmag[apo_idx:idxs[1]])
        ) + apo_idx'''
        #idxs = np.concatenate((idxs, [_idx]))
    
    '''idx_100pc = cmf.general.get_idx_in_array(
        0.1, 
        sep[np.r_[apo_idx:idxs[1]]]
    ) + apo_idx
    idxs = np.concatenate((idxs, [idx_100pc]))'''

    idxs.sort()
    print(idxs)

    if True:
        try:
            bh1_bound, bh2_bound, merged = cmf.analysis.get_bound_binary(bhfile)
        except IndexError:
            print(f"Skipping file {i}")
            continue
        bh1_bound, bh2_bound = cmf.analysis.move_to_centre_of_mass(bh1_bound, bh2_bound)
        fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2,4,figsize=(8,5))
        #ax1 = plt.subplot(2,4,1)
        #ax2 = plt.subplot(2,4,2)
        #ax3 = plt.subplot(2,4,3)
        #ax4 = plt.subplot(2,4,4)
        #ax5 = plt.subplot(2,4,5)
        #ax6 = plt.subplot(2,4,6)
        #ax7 = plt.subplot(2,4,7)
        ax1.set_xlabel("t/Myr")
        ax2.set_xlabel("x/pc")
        ax2.set_ylabel("y/pc")
        ax3.set_xlabel("t/Myr")
        ax3.set_ylabel(r"v$_r$/km/s")
        ax4.set_xlabel("t/Myr")
        ax4.set_ylabel(r"v$_t$/km/s")
        ax5.set_xlabel("t/Myr")
        ax5.set_ylabel("b/pc")
        ax6.set_xlabel("t/Myr")
        ax6.set_ylabel(r"$\theta$")
        ax7.set_yscale("symlog", linthresh=0.5)
        fig = ax1.get_figure()
        fig.suptitle(f"{datadirs[datadir_num].split('/')[-2]}")
        if False:
            ax1.semilogy(bh1.t/myr, bmag/ketjugw.units.pc, markevery=[idxs], marker="o")
            ax1.set_ylabel("|b|/pc")
        else:
            ax1.semilogy(bh1.t/myr, sep/ketjugw.units.pc, markevery=[idxs], marker="x")
            ax1.set_ylabel("r/pc")
        ax5.plot(bh1.t/myr, bmag/ketjugw.units.pc, markevery=[idxs], marker="x")
        ax5.scatter(bh1.t[apo_idxs]/myr, bmag[apo_idxs]/ketjugw.units.pc, marker="o")
        ax5.axhline(np.nanmedian(bmag[apo_idxs[0]:idxs[1]])/ketjugw.units.pc, c="tab:red")
        ax6.plot(bh1.t/myr, theta/ketjugw.units.pc, markevery=[idxs], marker="x")
        ax6.scatter(bh1.t[apo_idxs]/myr, theta[apo_idxs]/ketjugw.units.pc, marker="o")
        bmagdiff = np.diff(bmag)/ketjugw.units.pc
        ax7.plot(bh1.t[1:]/myr, bmagdiff, markevery=[idxs], marker="x")
        ax7.scatter(bh1.t[apo_idxs]/myr, bmagdiff[apo_idxs], marker="o")
        
        #ax1.plot(y, markevery=[idxs], marker="o")
        for bh, bhb in zip((bh1, bh2), (bh1_bound, bh2_bound)):
            l = ax2.plot(bh.x[:,0]/ketjugw.units.pc, bh.x[:,2]/ketjugw.units.pc, alpha=0.2)
            #ax2.plot(bhb.x[:,0]/ketjugw.units.pc, bhb.x[:,2]/ketjugw.units.pc, color=l[0].get_color())
            ax2.scatter(bh.x[:,0][idxs]/ketjugw.units.pc, bh.x[:,2][idxs]/ketjugw.units.pc, marker="x")
            ax2.scatter(bh.x[:,0][apo_idxs]/ketjugw.units.pc, bh.x[:,2][apo_idxs]/ketjugw.units.pc, marker="o")
            #ax3.plot(bh.v[:,0]/ketjugw.units.km_per_s, bh.v[:,2]/ketjugw.units.km_per_s, alpha=0.2, markevery=[-1], marker="x")
            #ax3.scatter(bh.v[~mask,0]/ketjugw.units.km_per_s, bh.v[~mask,2]/ketjugw.units.km_per_s, marker=".", c=l[0].get_color())
            #ax3.scatter(bh.v[:,0][idxs]/ketjugw.units.km_per_s, bh.v[:,2][idxs]/ketjugw.units.km_per_s, marker="x")
        bh1v_s = cmf.mathematics.spherical_components(bh1.x/ketjugw.units.pc, bh1.v/ketjugw.units.km_per_s)
        bh2v_s = cmf.mathematics.spherical_components(bh2.x/ketjugw.units.pc, bh2.v/ketjugw.units.km_per_s)
        bh1v_t = np.sqrt(bh1v_s[:,1]**2 + bh1v_s[:,2]**2)
        bh2v_t = np.sqrt(bh2v_s[:,1]**2 + bh2v_s[:,2]**2)
        for bhvr, bhvt, bh in zip((bh1v_s[:,0], bh2v_s[:,0]), (bh1v_t, bh2v_t), (bh1, bh2)):
            hundred_kms_idx = cmf.general.get_idx_in_array(rv_crit, bhvr[apo_idxs[0]:idxs[2]])+apo_idxs[0]
            print(f"{rv_crit}km/s idx: {hundred_kms_idx}")
            ax3.plot(bh1.t/myr, bhvr, markevery=[idxs], marker="x")
            ax3.scatter(bh1.t[apo_idxs]/myr, bhvr[apo_idxs], marker="o")
            ax3.scatter(bh1.t[hundred_kms_idx]/myr, bhvr[hundred_kms_idx])
            ax4.plot(bh1.t/myr, bhvt, markevery=[idxs], marker="x")
            ax4.scatter(bh1.t[apo_idxs]/myr, bhvt[apo_idxs], marker="o")
            ax2.scatter(bh.x[hundred_kms_idx, 0]/ketjugw.units.pc, bh.x[hundred_kms_idx, 2]/ketjugw.units.pc, marker="^")
            ax1.scatter(bh.t[hundred_kms_idx]/myr, sep[hundred_kms_idx]/ketjugw.units.pc)
        ax5.scatter(bh1.t[hundred_kms_idx]/myr, bmag[hundred_kms_idx]/ketjugw.units.pc, marker="^")
        m, iqr = cmf.mathematics.quantiles_relative_to_median(bmag[hundred_kms_idx-100:hundred_kms_idx+100]/ketjugw.units.pc)
        median_b[i] = m
        iqr_b[0,i], iqr_b[1,i] = iqr
        op = ketjugw.orbital_parameters(bh1_bound, bh2_bound)
        m, iqr = cmf.mathematics.quantiles_relative_to_median(op["e_t"])
        median_eccs[i] = m
        iqr_eccs[0,i], iqr_eccs[1,i] = iqr
        plt.close()

    print("b values: ")
    for j in idxs:
        print(f"  -> idx {j}: {bmag[j]/ketjugw.units.pc:.3f}")
    print(f"  -> {rv_crit}km/s idx {hundred_kms_idx}: {bmag[hundred_kms_idx]/ketjugw.units.pc:.3f}")
    #plt.show()

print(iqr_b.shape)
print(iqr_b)

plt.errorbar(median_b, median_eccs, xerr=iqr_b, yerr=iqr_eccs, fmt=".")
plt.xlabel("b/pc")
plt.ylabel("e")
plt.ylim(0,1)
plt.title(f"{datadirs[datadir_num].split('/')[-2]}")
plt.show()