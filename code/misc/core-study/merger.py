import numpy as np
import matplotlib.pyplot as plt
import ketjugw
import baggins as bgs



#kfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/ketju_bhs_cp.hdf5"

kfiles = bgs.utils.get_ketjubhs_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/")

kpc = ketjugw.units.pc * 1e3
kms = ketjugw.units.km_per_s



if False:
    step = 10
    start_idx = 10000
    end_idx = 150000

    for i, kfile in enumerate(kfiles):
        if "960" in kfile: break
        if i%3!=0: continue
        print(kfile)
        bhs = ketjugw.load_hdf5(kfile)

        l = None
        for bh in bhs.values():
            _end_idx = min(end_idx, len(bh))
            l = plt.plot(bh.x[start_idx:_end_idx:step,0]/kpc, 
                        bh.x[start_idx:_end_idx:step, 2]/kpc, 
                        c=(l[-1].get_color() if l is not None else None)
                        )
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()
elif False:
    start_idx = -20
    end_idx = -1
    for i, kfile in enumerate(kfiles):
        if "960" in kfile: break
        if i%3!=0: continue
        print(kfile)
        bhs = ketjugw.load_hdf5(kfile)
        l = None
        bh1, bh2, *_ = bgs.analysis.get_bound_binary(kfile)
        bh1, bh2 = bgs.analysis.move_to_centre_of_mass(bh1,bh2)
        for bh in (bh1,bh2):
            l = plt.plot(bh.v[start_idx:end_idx,0]/kms,
                         bh.v[start_idx:end_idx,2]/kms,
                         c=(l[-1].get_color() if l is not None else None)
            )
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()
elif True:
    for i, kfile in enumerate(kfiles):
        if "960" in kfile: break
        if i%3!=0: continue
        print(kfile)
        bhs = ketjugw.load_hdf5(kfile)
        bh1, bh2, *_ = bgs.analysis.get_bound_binary(kfile)
        vcom = np.average(
            np.vstack((bh1.v[-1,:]/kms, bh2.v[-1,:]/kms)),
            weights = [bh1.m[-1], bh2.m[-1]],
            axis=0
        ).flatten()
        origin = np.atleast_2d(np.zeros(2))
        print(vcom)
        plt.quiver(*[0,0], vcom[0], vcom[2], angles="xy", scale_units="xy", scale=1)
        plt.xlim(-1.1*vcom[0], 1.1*vcom[0])
        plt.ylim(-1.1*vcom[2], 1.1*vcom[2])
    plt.show()
    quit()