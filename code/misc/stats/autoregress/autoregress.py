import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cm_functions as cmf



parser = argparse.ArgumentParser(description="Run stan model for Quinlan evolution.", allow_abbrev=False)
parser.add_argument(type=str, help="file of observed quantities", dest="obs_file")
args = parser.parse_args()


# set up Stan Model
my_stan = cmf.analysis.StanModel("stan/ar.stan", "stan/ar.stan", args.obs_file, figname_base="stats/autoregress/autoregress", autoregress=True)

# set up observed data
mask = my_stan.obs.loc[:, "name"] ==  np.unique(my_stan.obs.loc[:, "name"])[0]
data = dict(
    N_tot = np.sum(mask),
    observed_data = my_stan.obs.loc[mask, "e"]
)

my_stan.build_model()
my_stan.sample_model(data=data, save=False)
my_stan.posterior_plot("e", "e", "posterior_pred")

plt.show()