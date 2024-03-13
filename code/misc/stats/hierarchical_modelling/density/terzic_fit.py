import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs



my_stan = bgs.analysis.StanModel("stan/terzic_prior.stan", "stan/terzic_prior.stan", None, "stats/terzic_density")

data = dict(
    R = np.geomspace(1e-2, 1e2, 100),
    Upsilon = 4.0,
    N_child = 10,
    N_tot = 1000,
    N_per_child = 100,
    child_id = [i for i in range(10) for j in range(100)]
)

my_stan.sample_prior(data)

fig, ax = plt.subplots(1,1)
ax.set_xlabel("r/kpc")
ax.set_ylabel(r"$\Sigma(r)$/(M$_\odot$/kpc$^2$)")
my_stan.prior_plot(xmodel="R", ymodel="projected_density", ax=ax)

plt.show()