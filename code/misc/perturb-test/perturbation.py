import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ketjugw
import scipy.stats, scipy.interpolate
import cm_functions as cmf


def fit_path(tvals, ivals, tnorm=1, pnorm=1, nbins=50, com=[0,0,0]):
    vals = {'x':None, 'y':None, 'z':None}
    tvals = tvals / tnorm
    bin = vals.copy()
    interp = vals.copy()
    error = vals.copy()
    for i, key in enumerate(list(vals.keys())):
        vals[key] = (ivals[:,i]-com[:,i])/pnorm
        stat, binedges, binnum = scipy.stats.binned_statistic(tvals, vals[key], statistic='median', bins=nbins)
        bin[key] = {'stat':stat, 'binedges':binedges, 'binnum':binnum}
        midbins = (bin[key]['binedges'][1:] + bin[key]['binedges'][:-1]) / 2
        interp[key] = scipy.interpolate.interp1d(midbins, bin[key]['stat'], bounds_error=False, kind='cubic')
        error[key] = vals[key] - interp[key](tvals)
    return vals, bin, interp, error



bh_file = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x02/0-001/output/ketju_bhs.hdf5"
#bh_file = '/Volumes/Rawlings_Storage/KETJU/data/merger/P03_S03/ketju_bhs.hdf5'


kpc = ketjugw.units.pc * 1e3
myr = 1e6 * ketjugw.units.yr
kms = ketjugw.units.km_per_s
bhs = ketjugw.data_input.load_hdf5(bh_file)
bh1, bh2 = bhs.values()
#boundbinary = ketjugw.find_binaries(bhs, remove_unbound_gaps=False)

#bh1, bh2 = list(boundbinary.values())[0]
#params = ketjugw.orbital_parameters(bh1,bh2)

xcom = (bh1.x * bh1.m[0] + bh2.x * bh2.m[0]) / (bh1.m[0] + bh2.m[0])
vcom = (bh1.v * bh1.m[0] + bh2.v * bh2.m[0]) / (bh1.m[0] + bh2.m[0])


if True:
    saveDir = "/users/arawling/figures/perturb-test/"
    cols = cmf.plotting.mplColours()
    nbins = 50
    setbinwidth = [100, 50, 25, 20, 15, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05]
    endidx = 100100
    wander_radius_vec = np.full((len(setbinwidth),2), np.nan)
    brown_vel_vec = np.full((len(setbinwidth),2), np.nan)

    for sind, sbw in enumerate(setbinwidth):
        fig = plt.figure(constrained_layout=True, figsize=(12,6))
        gs = fig.add_gridspec(3, 4)
        axx1 = fig.add_subplot(gs[0, :2])
        axx2 = fig.add_subplot(gs[1, 0])
        axx3 = fig.add_subplot(gs[1, 1])
        axx4 = fig.add_subplot(gs[2, 0])
        axx5 = fig.add_subplot(gs[2, 1])
        axv1 = fig.add_subplot(gs[0, 2:])
        axv2 = fig.add_subplot(gs[1, 2])
        axv3 = fig.add_subplot(gs[1, 3])
        axv4 = fig.add_subplot(gs[2, 2])
        axv5 = fig.add_subplot(gs[2, 3])

        #set up labels
        axx1.set_xlabel('x/kpc')
        axx1.set_ylabel('z/kpc')
        axx1.set_title('Position')
        axx2.set_ylabel('BH 1')
        axx4.set_ylabel('BH 2')
        for axA, axB in ((axx2, axx3), (axx4, axx5)):
            axA.set_xlabel('Pos. Error [kpc]')
            axB.set_xlabel('t/Myr')
            axB.set_ylabel('Pos. Error [kpc]')
        axv1.set_xlabel(r'v$_\mathrm{x}$/km/s')
        axv1.set_ylabel(r'v$_\mathrm{z}$/km/s')
        axv1.set_title('Velocity')
        axv1.text(0.01, 0.8, 'bins: {:.2f} Myr'.format(sbw), transform=axv1.transAxes)
        for axA, axB in ((axv2, axv3), (axv4, axv5)):
            axA.set_xlabel('Vel. Error [km/s]')
            axB.set_xlabel('t/Myr')
            axB.set_ylabel('Vel. Error [km/s]')
        for axA in (axx3, axx5, axv3, axv5):
            axA.axhline(0, c='k', alpha=0.4, zorder=0)

        '''print(np.floor(bh1.t[:endidx]/myr / setbinwidth[0]))
        endidx = np.where(np.abs(bh1.t - np.floor(bh1.t[:endidx] / setbinwidth[0])[0])< 1e-5)[0]
        print(endidx)
        quit()'''
        errors = {'0':{'pos':None, 'vel':None}, '1':{'pos':None, 'vel':None}}
        print('Analysis performed to {:.2f} Myr'.format(bh1.t[endidx]/myr))

        for ind, bh in enumerate((bh1, bh2)):
            set_bins = np.arange(bh.t[0]/myr, bh.t[endidx]/myr, sbw)
            print('Bin width: {} Myr'.format(set_bins[1]-set_bins[0]))
            posvals, posbin, posinterp, poserror = fit_path(bh.t[:endidx], bh.x[:endidx], tnorm=myr, pnorm=kpc, nbins=set_bins, com=xcom[:endidx])
            errors[str(ind)]['pos'] = poserror
            velvals, velbin, velinterp, velerror = fit_path(bh.t[:endidx], bh.v[:endidx], tnorm=myr, pnorm=kms, nbins=set_bins, com=vcom[:endidx])
            errors[str(ind)]['vel'] = velerror

            axx1.plot(posvals['x'], posvals['z'], alpha=0.7, color=cols[ind])
            axx1.plot(posbin['x']['stat'], posbin['z']['stat'], '-o', zorder=10, color=cols[ind], lw=0.3)
            axv1.plot(velvals['x'], velvals['z'], alpha=0.7, color=cols[ind])
            axv1.plot(velbin['x']['stat'], velbin['z']['stat'], '-o', zorder=10, color=cols[ind], lw=0.3)
            for i, key in enumerate(list(poserror.keys())):
                if ind == 0:
                    axx2.hist(poserror[key], nbins, alpha=0.7, density=True, histtype='step')
                    axx3.plot(bh.t[:endidx]/myr, poserror[key], alpha=0.7)
                    axv2.hist(velerror[key], nbins, alpha=0.7, density=True, histtype='step')
                    axv3.plot(bh.t[:endidx]/myr, velerror[key], alpha=0.7)
                else:
                    axx4.hist(poserror[key], nbins, alpha=0.7, density=True, histtype='step')
                    axx5.plot(bh.t[:endidx]/myr, poserror[key], alpha=0.7)
                    axv4.hist(velerror[key], nbins, alpha=0.7, density=True, histtype='step')
                    axv5.plot(bh.t[:endidx]/myr, velerror[key], alpha=0.7)
            maxidx = np.argmax(np.abs(poserror['z']))
            print('Time of peak positional error (BH{}): {:.2f}'.format(ind, bh.t[maxidx]/myr))
            axx1.scatter(posvals['x'][maxidx], posvals['z'][maxidx], marker='*', c=cols[ind], zorder=20, label=('Peak Error' if ind==0 else ''))
            axv1.scatter(velvals['x'][maxidx], velvals['z'][maxidx], marker='*', c=cols[ind], zorder=20)
        axx1.legend()


        for ind in range(2):
            print('BH{}'.format(ind))
            #get the positional error
            pos_var = {'x':0, 'y':0, 'z':0}
            vel_var = pos_var.copy()
            for ind2, crd in enumerate(list(pos_var.keys())):
                eps = 0.01
                mask = np.logical_not(np.isnan(errors[str(ind)]['pos'][crd])) #np.logical_and(errors[str(ind)]['pos'][crd] > np.quantile(errors[str(ind)]['pos'][crd], eps), errors[str(ind)]['pos'][crd] < np.quantile(errors[str(ind)]['pos'][crd], 1-eps))
                posfitparams = scipy.stats.gennorm.fit(errors[str(ind)]['pos'][crd][mask])
                posfit = scipy.stats.gennorm(*posfitparams)
                ppflim = 1e-3
                xs = np.linspace(posfit.ppf(ppflim), posfit.ppf(1-ppflim), 250)
                ax = axx2 if not ind else axx4
                ax.plot(xs, posfit.pdf(xs), color=cols[ind2], alpha=0.5)
                pos_var[crd] = posfit.moment(2) #np.nanstd(errors[str(ind)]['pos'][crd])**2
                mask = np.logical_not(np.isnan(errors[str(ind)]['vel'][crd]))
                velfitparams = scipy.stats.gennorm.fit(errors[str(ind)]['vel'][crd][mask])
                velfit = scipy.stats.gennorm(*velfitparams)
                vs = np.linspace(velfit.ppf(ppflim), velfit.ppf(1-ppflim), 250)
                ax = axv2 if not ind else axv4
                ax.plot(vs, velfit.pdf(vs), color=cols[ind2], alpha=0.5)
                vel_var[crd] = velfit.moment(2) #np.nanstd(errors[str(ind)]['vel'][crd])**2
            wander_radius = np.sqrt(sum(v for v in pos_var.values()))
            brown_vel = np.sqrt(sum(v for v in vel_var.values()))
            wander_radius_vec[sind, ind] = wander_radius
            brown_vel_vec[sind, ind] = brown_vel
            print(' Wandering radius: {:.3e} kpc'.format(wander_radius))
            print(' Brownian velocity: {:.3e} km/s'.format(brown_vel))
        plt.savefig('{}{}.png'.format(saveDir, sbw), dpi=300)
        plt.close()
    fig, ax = plt.subplots(1,2, figsize=(7,4), sharex='all')
    ax[0].set_xscale('log')
    #ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    ax[0].set_xlabel('Bin width [Myr]')
    ax[0].set_ylabel('Wandering radius [kpc]')
    ax[1].set_xlabel('Bin width [Myr]')
    ax[1].set_ylabel('Brownian velocity [km/s]')
    ax[0].plot(setbinwidth, wander_radius_vec, '-o')
    ax[1].plot(setbinwidth, brown_vel_vec, '-o')
    plt.tight_layout()
    plt.savefig('{}converge.png'.format(saveDir), dpi=300)
    plt.close()




if False:
    fig, ax = plt.subplots(1, 2, figsize=(7,3), subplot_kw={'projection':'3d'})
    for axi in ax:
        axi.set_xlabel('x')
        axi.set_ylabel('y')
        axi.set_zlabel('z')
    for bh in (bh1, bh2):
        #ax[0].plot(bh.x[:,0]/kpc, bh.x[:,1]/kpc, bh.x[:,2]/kpc)
        ax[0].plot((bh.x[:idx,0]-xcom[:idx,0])/kpc, (bh.x[:idx,1]-xcom[:idx,1])/kpc, (bh.x[:idx,2]-xcom[:idx,2])/kpc)
        ax[0].scatter(bh.x[idx,0]/kpc, bh.x[idx,1]/kpc, bh.x[idx,2]/kpc, zorder=10, marker='o', s=80)
        ax[1].plot(bh.v[:idx:step,0]-vcom[:idx:step,0], bh.v[:idx:step,1]-vcom[:idx:step,1], bh.v[:idx:step,2]-vcom[:idx:step,2])
        ax[1].scatter(bh.v[idx,0]-vcom[-1,0], bh.v[idx,1]-vcom[idx,1], bh.v[idx,2]-vcom[idx,2], zorder=10, marker='o', s=80)
    #ax[1].plot((bh1.x[:,0]-bh2.x[:,0])/kpc, (bh1.x[:,1]-bh2.x[:,1])/kpc, (bh1.x[:,2]-bh2.x[:,2])/kpc)
    plt.show()
    quit()
