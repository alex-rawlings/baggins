import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baggins as bgs


parser = argparse.ArgumentParser(description="Run stan model for Quinlan evolution.", allow_abbrev=False)
parser.add_argument(type=str, help="file of observed quantities", dest="obs_file")
args = parser.parse_args()

df = pd.read_pickle(args.obs_file)

print(df)

def insetax(ax, x, y, bounds, xlim, ylim):
     # set up the inset axis
    axins = ax.inset_axes(bounds)
    ax.scatter(x, y, marker=".")
    axins.scatter(x, y, marker=".")
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    ax.indicate_inset_zoom(axins, linewidth=2)


fig, ax = plt.subplots(1,2)
ax[0].set_xlabel("t/Myr")
ax[0].set_ylabel("a/pc")
ax[1].set_xlabel(r"a$_{i-1}$/pc")
ax[1].set_ylabel(r"a$_{i}$/pc")
#ax[1].set_xlabel("Count/bin")
#ax[1].set_ylabel(r"$\sqrt{\mathrm{E}(\sigma_e^2)}$")
#ax[1].set_xscale("log")
#ax[1].set_yscale("log")

chars = bgs.plotting.mplChars()

for j, n in enumerate(np.unique(df.loc[:, "name"])):
    if j>0: break
    print(f"Child {n}")
    mask = df.loc[:, "name"] == n
    x_data = df.loc[mask, "t"].to_numpy()
    y_data = 1/df.loc[mask, "a"].to_numpy()
    if True:
        insetax(ax[0], x_data, y_data, [0.5, 0.05, 0.3, 0.3], [402, 405], [0.0405, 0.042])
        insetax(ax[1], y_data[:-1], y_data[1:], [0.5, 0.05, 0.3, 0.3], [0.04, 0.041], [0.04, 0.041])
    else:
        s = ax[0].scatter(x_data, y_data, marker=".")
        ax[1].scatter(y_data[:-1], y_data[1:], marker=".")
    '''for char, p in zip(chars, (2, 5, 10, 100, 1000, 5000, 10000, 50000)):
        l = int(np.floor(len(x_data)/p))
        x_tmp = np.full(l, np.nan, dtype=float)
        xerr_tmp = np.full_like(x_tmp, np.nan)
        y_tmp = np.full_like(x_tmp, np.nan)
        yerr_tmp = np.full_like(x_tmp, np.nan)
        for i in range(l):
            idx = np.r_[i*p:(i+1)*p]
            x_tmp[i] = np.nanmean(x_data[idx])
            xerr_tmp[i] = np.nanstd(x_data[idx])
            y_tmp[i] = np.nanmean(y_data[idx])
            yerr_tmp[i] = np.nanstd(y_data[idx])
        #ax[0].errorbar(x_tmp, y_tmp, xerr=xerr_tmp, yerr=yerr_tmp, fmt=char, label=(p if j==0 else ""), c=s.get_facecolor())
        ax[1].scatter(p, np.sqrt(np.nanmean(yerr_tmp**2)), c=s.get_facecolor())'''

#ax[0].legend(loc="upper right", title="Count/bin")
plt.show()