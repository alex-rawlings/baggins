import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pygad
import cm_functions as cmf


#set up the command line arguments
parser = argparse.ArgumentParser(description="Plot and compare beta(r) profiles for the given snapshots.", allow_abbrev=False)
parser.add_argument(type=str, help="Common path of all snapshots to compare", dest="path")
parser.add_argument("-f", "--file", type=str, help="path to snapshot relative to common path", dest="files", action="append")
parser.add_argument("-a", "--all", help="analyse all snapshots in path", dest="all", action="store_true")
parser.add_argument("-bw", "--binwdith", type=float, help="fixed bin width to determine beta within", dest="binwidth", default=0.05)
parser.add_argument("-om", "--OsipkovMerritt", dest="osipkovmerritt", action="store_true", help="fit an Osipkov-Merritt model to the data")
parser.add_argument("-fp", "--figpath", type=str, help="figure directory", dest="figpath", default="{}/beta/".format(cmf.FIGDIR))
parser.add_argument("-fn", "--figname", type=str, help="figure name", dest="figname", default="beta")
parser.add_argument('-S', '--singlesystem', dest='singlesystem', action='store_true', help='analyse the snapshots as a single system')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose printing in script')
args = parser.parse_args()

#quantile cut for plotting purposes only
plot_quantile_min = 0.05
if args.osipkovmerritt:
    num_rows = 2
    gridspec = {"height_ratios":[3,1]}
else:
    num_rows = 1
    gridspec = {}

if args.all:
    snapfiles = cmf.utils.get_snapshots_in_dir(args.path)
else:
    snapfiles = [os.path.join(args.path, p) for p in args.files]

for i, snapfile in enumerate(snapfiles):
    if args.verbose:
        print("Reading: {}".format(snapfile))
    snap = pygad.Snapshot(snapfile)
    snap.to_physical_units()
    snaptime = cmf.general.convert_gadget_time(snap)
    if i == 0:
        number_of_systems = len(snap.bh["ID"])
        snaptimes = np.full_like(snapfiles, np.nan, dtype=float)
        ra_values = np.full((len(snapfiles), number_of_systems), np.nan, dtype=float)
        if args.singlesystem or number_of_systems == 1:
            fig, ax = plt.subplots(num_rows,1, sharex="col", squeeze=False, gridspec_kw=gridspec)
            if not isinstance(ax, np.ndarray):
                ax = [ax]
            args.singlesystem = True
            star_id_masks = None
        else:
            fig, ax = plt.subplots(num_rows, number_of_systems, sharex="all", sharey="row", squeeze=False, gridspec_kw=gridspec)
            star_id_masks = cmf.analysis.get_all_id_masks(snap)
        lower_beta = 100
    snaptimes[i] = snaptime

    #recentre
    xcom = cmf.analysis.get_com_of_each_galaxy(snap, masks=star_id_masks, verbose=args.verbose)
    vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom=xcom, masks=star_id_masks, verbose=args.verbose)
    for ind, idx in enumerate(snap.bh["ID"]):
        #move to this system's com before masking
        snap.stars["pos"] -= xcom[idx]
        snap.stars["vel"] -= vcom[idx]
        if args.singlesystem:
            subsnap = snap.stars
        else:
            subsnap = snap.stars[star_id_masks[idx]]
        #determine spherical velocity and beta
        vr, vtheta, vphi = cmf.mathematics.spherical_components(subsnap["pos"], subsnap["vel"])
        beta_r, radbins, bincount = cmf.analysis.beta_profile(subsnap["r"], vr, vtheta, vphi, args.binwidth)
        this_lower_beta = np.quantile(beta_r, plot_quantile_min)
        if this_lower_beta < lower_beta:
            lower_beta = this_lower_beta
        if args.osipkovmerritt:
            #fit the data with the theoretical expectation
            #weight the data points by the inverse of the bin count
            bincount[bincount<1] = 1
            sigma = 1/np.sqrt(bincount)
            params_opt, params_cov = scipy.optimize.curve_fit(cmf.literature.OsipkovMerritt, radbins, beta_r, p0=[1], bounds=(0, 100), sigma=sigma)
            ra_values[i][ind] = params_opt[0]
        labsize = bincount/np.max(bincount) * 20
        if ind == 0:
            s = ax[0][ind].scatter(radbins, beta_r, s=labsize, label="t={:.3f} Gyr".format(snaptime), zorder=10)
        else:
            ax[0][ind].scatter(radbins, beta_r, s=labsize, c=s.get_facecolor(), zorder=10+ind)
        if args.osipkovmerritt:
            om_values = cmf.literature.OsipkovMerritt(radbins, *params_opt)
            ax[0][ind].plot(radbins, om_values, alpha=0.8, zorder=1, label=r"r$_a$: {:.2f}".format(params_opt[0]))
            ax[1][ind].loglog(radbins, np.abs(beta_r-om_values), zorder=10+ind)
        if args.singlesystem:
            break
        else:
            ax[0][ind].set_title("BH: {}".format(idx))
        #move back to global coordinate before next iteration
        snap.stars["pos"] += xcom[idx]
        snap.stars["vel"] += vcom[idx]
        snap.delete_blocks()

ax[0][0].legend()
if args.verbose:
    print("Plot will have lower y-lim truncated to the lowest {} quantile value determined from all input datasets".format(plot_quantile_min))
ax[0][0].set_ylim(lower_beta, 1.1)
ax[0][0].set_xscale("log")
ax[0][0].set_ylabel(r"$\beta(r)$")
fig.suptitle("{}".format(args.figname))
for axi in ax[1 if args.osipkovmerritt else 0]:
    axi.set_xlabel("r/kpc")
if args.osipkovmerritt:
    ax[1][0].set_ylabel(r"$|\beta_\mathrm{obs} - \beta_\mathrm{th}|$")
os.makedirs(args.figpath, exist_ok=True)
fig.savefig(os.path.join(args.figpath, args.figname))

if args.osipkovmerritt:
    print(ra_values, snaptimes)
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(snaptimes, ra_values)
    ax2.set_xlabel("t/Gyr")
    ax2.set_ylabel("ra/kpc")
    fig2.savefig(os.path.join(args.figpath, "{}-times".format(args.figname)))
plt.show()
