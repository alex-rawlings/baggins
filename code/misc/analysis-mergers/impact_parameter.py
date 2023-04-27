import os.path
import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import cm_functions as cmf



datadirs = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_05",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_07",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_09",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/nasim/gualandris/stars_only_e_099"
]

bound_idxs = 20
myr = 1e6 * ketjugw.units.yr

bhfiles = cmf.utils.get_ketjubhs_in_dir(datadirs[3])

for i, bhfile in enumerate(bhfiles):
    bh1, bh2, merged = cmf.analysis.get_bh_particles(bhfile)
    bh1, bh2 = cmf.analysis.move_to_centre_of_mass(bh1, bh2)

    energy = ketjugw.orbital_energy(bh1, bh2)
    mask = energy>0

    b = cmf.analysis.impact_parameter(bh1, bh2)
    bmag = cmf.mathematics.radial_separation(b)
    #y = np.diff(np.sign(np.diff(bmag[mask], prepend=0)), prepend=0)
    #idxs = np.where(y==-2)[0][-1]-1
    #idxs = cmf.general.get_idx_in_array(23.9, bh1.t[mask]/myr)

    _, idxs, sep = cmf.analysis.find_pericentre_time(bh1[mask], bh2[mask], return_sep=True, prominence=0.01)
    print(idxs)
    idxs = idxs[:2]

    apo_idx = np.argmax(sep[np.r_[idxs[0]:idxs[1]]]) + idxs[0]
    idxs = np.concatenate((idxs, [apo_idx]))
    #print(apo_idx)

    '''half_apo_peri_idx = cmf.general.get_idx_in_array(
        (sep[apo_idx] + sep[idxs[1]])/2,
        sep[np.r_[apo_idx:idxs[1]]]
    ) + apo_idx'''

    for q in (0.5, 0.25, 0.10, 0.05, 0.01):
        _idx = cmf.general.get_idx_in_array(
            np.nanquantile(np.log10(bmag[apo_idx:idxs[1]]), q),
            np.log10(bmag[apo_idx:idxs[1]])
        ) + apo_idx
        idxs = np.concatenate((idxs, [_idx]))
    
    '''idx_100pc = cmf.general.get_idx_in_array(
        0.1, 
        sep[np.r_[apo_idx:idxs[1]]]
    ) + apo_idx
    idxs = np.concatenate((idxs, [idx_100pc]))'''

    idxs.sort()
    print(idxs)

    if i==1:
        bh1_bound, bh2_bound, merged = cmf.analysis.get_bound_binary(bhfile)
        bh1_bound, bh2_bound = cmf.analysis.move_to_centre_of_mass(bh1_bound, bh2_bound)
        #fig, (ax1,ax2) = plt.subplots(2,1)
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
        ax1.set_xlabel("t/Myr")
        ax2.set_xlabel("x/pc")
        ax2.set_ylabel("y/pc")
        ax3.set_xlabel("t/Myr")
        ax3.set_ylabel(r"v$_r$/km/s")
        ax4.set_xlabel("t/Myr")
        ax4.set_ylabel(r"v$_t$/km/s")
        if False:
            ax1.semilogy(bh1.t[mask]/myr, bmag[mask]/ketjugw.units.pc, markevery=[idxs], marker="o")
            ax1.set_ylabel("|b|/pc")
        else:
            ax1.semilogy(bh1.t[mask]/myr, sep*1e3, markevery=[idxs], marker="o")
            ax1.set_ylabel("r/pc")
        #ax1.plot(y, markevery=[idxs], marker="o")
        for bh, bhb in zip((bh1, bh2), (bh1_bound, bh2_bound)):
            l = ax2.plot(bh.x[:,0]/ketjugw.units.pc, bh.x[:,2]/ketjugw.units.pc, alpha=0.2)
            ax2.plot(bhb.x[:,0]/ketjugw.units.pc, bhb.x[:,2]/ketjugw.units.pc, color=l[0].get_color())
            ax2.scatter(bh.x[mask,0][idxs]/ketjugw.units.pc, bh.x[mask,2][idxs]/ketjugw.units.pc, marker="x")
            #ax3.plot(bh.v[mask,0]/ketjugw.units.km_per_s, bh.v[mask,2]/ketjugw.units.km_per_s, alpha=0.2, markevery=[-1], marker="x")
            #ax3.scatter(bh.v[~mask,0]/ketjugw.units.km_per_s, bh.v[~mask,2]/ketjugw.units.km_per_s, marker=".", c=l[0].get_color())
            #ax3.scatter(bh.v[mask,0][idxs]/ketjugw.units.km_per_s, bh.v[mask,2][idxs]/ketjugw.units.km_per_s, marker="x")
        bh1v_s = cmf.mathematics.spherical_components(bh1.x/ketjugw.units.pc, bh1.v/ketjugw.units.km_per_s)
        bh2v_s = cmf.mathematics.spherical_components(bh2.x/ketjugw.units.pc, bh2.v/ketjugw.units.km_per_s)
        bh1v_t = np.sqrt(bh1v_s[:,1]**2 + bh1v_s[:,2]**2)
        bh2v_t = np.sqrt(bh2v_s[:,1]**2 + bh2v_s[:,2]**2)
        for bhvr, bhvt in zip((bh1v_s[:,0], bh2v_s[:,0]), (bh1v_t, bh2v_t)):
            ax3.plot(bh1.t[mask]/myr, bhvr[mask], markevery=[idxs], marker="x")
            ax4.plot(bh1.t[mask]/myr, bhvt[mask], markevery=[idxs], marker="x")
    print("b values: ")
    for j in idxs:
        print(f"  -> idx {j}: {bmag[mask][j]/ketjugw.units.pc:.3f}")
plt.show()