import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cm_functions as cmf



my_stan = cmf.analysis.StanModel(model_file="stan/graham_prior.stan", prior_file="stan/graham_prior.stan", obs_file=None, figname_base="stats/graham_density/graham_density")
N = 1000

data = dict(
    N_tot = N,
    R = np.geomspace(1e-2, 1e2, N),
    a = 10.0,
)

obs_file = "/scratch/pjohanss/arawling/testing/alex_density.pickle"
observations = cmf.utils.load_data(obs_file)
my_stan.obs = pd.DataFrame(data={"r":observations["run2"]["x"], "surf_rho":np.log10(np.median(observations["run2"]["density"]["0.0"], axis=0)), "group":["AD" for _ in range(len(observations["run2"]["x"]))]})
my_stan.categorical_label = "group"

my_stan.sample_prior(data)

fig, ax = plt.subplots(1,1)
ax.set_ylim(-1, 15.1)
ax.set_xlabel("r/kpc")
ax.set_ylabel(r"log($\Sigma(r)$/(M$_\odot$/kpc$^2$))")
ax.set_xscale("log")
#ax.set_yscale("log")
my_stan.prior_plot("r", "surf_rho", xmodel="R", ymodel="projected_density", ax=ax)

plt.show()