import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft
import baggins as bgs



parser = argparse.ArgumentParser(description="Run stan model for Quinlan evolution.", allow_abbrev=False)
parser.add_argument(type=str, help="file of observed quantities", dest="obs_file")
parser.add_argument(type=str, help="quantity to autoregress", dest="var")
args = parser.parse_args()


slope = 0.9994765
intercept = 0.0004847295
scatter = 0.0004624485
step = 500

rng = np.random.default_rng(42)

def autofunc(y):
    return intercept + slope * y# + rng.normal(0, scatter)

# set up pandas data frame to hold autoregression values
df = pd.read_pickle(args.obs_file)
df["inv_a"] = 1.0 / df.loc[:, "a"]

names =  np.unique(df.loc[:, "name"])

for n in names:
    mask = df.loc[:, "name"] == n
    #plt.plot(df.loc[mask, "t"], df.loc[mask, args.var], marker=".")
    x = pd.plotting.autocorrelation_plot(np.diff(df.loc[mask, args.var].to_numpy()))
    x.plot()
    break
    tot = np.sum(mask)
    d = df.loc[mask, "t"][1] - df.loc[mask, "t"][0]
    ft = np.real(scipy.fft.fftfreq(tot, d))
    a_t = np.real(scipy.fft.fft(df.loc[mask, args.var].to_numpy()))
    plt.scatter(ft, a_t)
    break
    i = 0
    t = []
    a = []
    t.append(df.loc[i, "t"])
    a.append(autofunc(df.loc[i, "inv_a"]))
    while i*step<tot:
        t.append(df.loc[i*step, "t"])
        #a.append(autofunc(a[i]))
        a.append(df.loc[i*step, "inv_a"])
        i += 1
    break
#print(len(t), len(a))
#plt.scatter(t, a, c="tab:red")

plt.show()