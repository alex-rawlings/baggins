import argparse
import os.path
import numpy as np
import scipy.stats, scipy.spatial.transform
import matplotlib.pyplot as plt
import cm_functions as cmf


def plothist_and_best(mu, sigma, ax1, ax2, ax3, nbins=10, label=None, endpoint=1e-3, print_label=None):
    """
    Plot the histogram of means and standard deviations, and the Gaussian 
    distribution described by the optimum of these parameters

    Parameters
    ----------
    mu: array of means
    sigma: array of standard deviations
    ax1,ax2,ax3: pyplot axes to plot to
    nbins: number of bins for histogram
    label: plot label
    endpoint: plotting domain (parsed to the ppf method of scipy distribution 
              object)
    """
    ax1.hist(mu, nbins, histtype="step", density=True)
    h,bins,_ = ax2.hist(sigma, nbins, histtype="step", density=True, label=label)
    best_mean = np.nanmean(mu)
    bincentres = cmf.mathematics.get_histogram_bin_centres(bins)
    best_sd = bincentres[np.argmax(h)]
    print("{}: mean={:.2e}, sd={:.2e}".format(print_label, best_mean, best_sd))
    distr = scipy.stats.norm(best_mean, best_sd)
    x = np.linspace(distr.ppf(endpoint), distr.ppf(1-endpoint), 100)
    ax3.plot(x, distr.pdf(x), label=label)


parser = argparse.ArgumentParser(description="Randomly rotate offset measurements to determine the best fit Gaussian distribution", allow_abbrev=False)
parser.add_argument(type=str, help="path to .pickle files", dest="path")
parser.add_argument("-f", "--file", type=str, help=".pickle file of Brownian offsets", dest="data", action="append")
parser.add_argument("-e", "--endpoints", type=float, help="plotting range of distributions", dest="endpoints", default=0.01)
parser.add_argument("-r", "--rotations", type=int, dest="rotations", help="number of random rotations to perform", default=5000)
parser.add_argument("-b", "--bins", type=int, help="number of histogram bins", dest="bins", default=20)
parser.add_argument("-t", "--time", type=float, help="Only use data for times above this value in the fits", dest="timelower", default=4.0)
args = parser.parse_args()

#shortcut method to get all files in a directory
if args.data[0] == "all":
    args.data = cmf.utils.get_files_in_dir(args.path, ext=".pickle", name_only=True)

fig, ax = plt.subplots(3,4, sharex="col", sharey="col", figsize=(8,7))
ax[0,0].set_title("Position Mean")
ax[0,1].set_title("Position SD")
ax[0,2].set_title("Velocity Mean")
ax[0,3].set_title("Velocity SD")

fig2, ax2 = plt.subplots(3,2, sharex="col", sharey="col")
ax2[0,0].set_title("Position")
ax2[0,1].set_title("Velocity")

for i, val in enumerate(("x", "y", "z")):
    ax[i,0].set_xlabel(val)
    ax[i,1].set_xlabel(val)
    ax[i,2].set_xlabel(r"v$_{}$".format(val))
    ax[i,3].set_xlabel(r"v$_{}$".format(val))
    ax2[i,0].set_xlabel(val)
    ax2[i,1].set_xlabel(r"v$_{}$".format(val))

for l, datafile in enumerate(args.data):
    data_dict = cmf.utils.load_data(os.path.join(args.path, datafile))
    time_mask = data_dict["times"] > args.timelower
    means = dict(
        pos = np.full((args.rotations, 3), np.nan),
        vel = np.full((args.rotations,3), np.nan)
    )
    sds = dict(
        pos = np.full((args.rotations, 3), np.nan),
        vel = np.full((args.rotations,3), np.nan)
    )
    for j in range(args.rotations):
        print("Rotating {:.1f}%              ".format(j/(args.rotations-1)*100), end="\r")
        rotation = scipy.spatial.transform.Rotation.random()
        pos = rotation.apply(data_dict["diff_x"][time_mask])
        vel = rotation.apply(data_dict["diff_v"][time_mask])
        for k in range(3):
            means["pos"][j,k], sds["pos"][j,k] = scipy.stats.norm.fit(pos[:,k])
            means["vel"][j,k], sds["vel"][j,k] = scipy.stats.norm.fit(vel[:,k])
    print("Rotation complete                                      ")

    for i in range(3):
        plothist_and_best(means["pos"][:,i], sds["pos"][:,i], ax[i,0], ax[i,1], ax2[i,0], nbins=args.bins, label=data_dict["galaxy_name"], endpoint=args.endpoints, print_label="{} Pos ({})".format(data_dict["galaxy_name"], i))
        plothist_and_best(means["vel"][:,i], sds["vel"][:,i], ax[i,2], ax[i,3], ax2[i,1], nbins=args.bins, endpoint=args.endpoints, print_label="{} Vel ({})".format(data_dict["galaxy_name"], i))
ax[0,1].legend()
ax2[0,0].legend()
fig.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/random_rots.png"))
fig2.savefig(os.path.join(cmf.FIGDIR, "brownian/along_axes/bestfit_from_random.png"))
plt.show()