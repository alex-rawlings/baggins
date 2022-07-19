import argparse
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cm_functions as cmf



parser = argparse.ArgumentParser(description="Run stan model for Quinlan evolution.", allow_abbrev=False)
parser.add_argument(type=str, help="file of observed quantities", dest="obs_file")
parser.add_argument(type=str, help="quantity to autoregress", dest="var")
args = parser.parse_args()

# set up pandas data frame to hold autoregression values
autoregress_df = pd.DataFrame(columns=["name", "slope", "intercept", "scatter"])

# set up Stan Model
my_stan = cmf.analysis.StanModel("stan/ar.stan", "stan/ar.stan", args.obs_file, figname_base=f"stats/autoregress/{args.var}", autoregress=True)#, random_select_obs={"num":200, "group":"name"})

# do data transformations
my_stan.transform_obs("a", "inv_a", lambda a:1/a)

names =  np.unique(my_stan.obs.loc[:, "name"])

for n in names:
    # set up observed data
    print(f"Group: {n}")
    my_stan.observation_mask = my_stan.obs.loc[:, "name"] == n
    data = dict(
        N_tot = np.sum(my_stan.observation_mask),
        observed_data = my_stan.obs.loc[my_stan.observation_mask, args.var]
    )
    my_stan.figname_base = f"{my_stan.figname_base}_{n}"
    my_stan.build_model()
    my_stan.sample_model(data=data, save=False, sample_kwargs={"output_dir":os.path.join(cmf.env_config.data_dir, "stan_files")})
    my_stan.parameter_plot(["alpha", "beta", "sigma"])
    my_stan.posterior_plot(args.var, args.var, "posterior_pred")
    plt.close()

    autoregress_df = pd.concat([autoregress_df, pd.DataFrame.from_dict({"name":[n], "slope":[np.nanmedian(my_stan._fit.stan_variable("beta"))], "intercept":[np.nanmedian(my_stan._fit.stan_variable("alpha"))], "scatter":[np.nanmedian(my_stan._fit.stan_variable("sigma"))]})])
    my_stan.figname_base = my_stan.figname_base.rstrip(f"_{n}")

#plt.show()
print(autoregress_df)

autoregress_df.to_csv(f"{args.var}_autoregress.csv", index=False)