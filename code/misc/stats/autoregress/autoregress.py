import argparse
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import baggins as bgs

bgs.plotting.check_backend()

def generate_mock_data(G=6, N_per_group=20, seed=42):
    np.random.seed(seed)
    N_total = G * N_per_group
    group_id = np.repeat(np.arange(1, G + 1), N_per_group)
    np.random.shuffle(group_id)


    x = np.linspace(0, 10, N_total)
    f = np.zeros(N_total)
    y = np.zeros(N_total)

    for g in range(1, G + 1):
        idx = np.where(group_id == g)[0]
        xg = x[idx]
        lengthscale = np.random.uniform(0.5, 1.5)
        sigma_f = np.random.uniform(0.5, 1.0)
        K = sigma_f**2 * np.exp(-0.5 * (np.subtract.outer(xg, xg)**2) / lengthscale**2)
        K += np.eye(N_per_group) * 1e-6
        f[idx] = np.random.multivariate_normal(np.zeros(N_per_group), K)
        y[idx] = f[idx] + np.random.normal(0, 0.1, N_per_group)

    return {
        'N_groups': G,
        #'N': [N_per_group] * G,
        'N_tot': N_total,
        'x': x,
        'y': y,
        'ids': group_id.astype(int)
    }

def compile_and_run_stan(stan_file_path, data_dict):
    model = CmdStanModel(stan_file=stan_file_path)
    fit = model.sample(data=data_dict, chains=4, parallel_chains=4)
    return fit

def plot_posterior_predictive(fit, data_dict):
    y_rep = fit.stan_variable('posterior_pred')
    x = data_dict['x']
    y = data_dict['y']
    id = data_dict['group_id']
    cmapper, sm = bgs.plotting.create_normed_colours(0, max(id), cmap="flare")

    mean_pred = np.nanmean(y_rep, axis=0)
    lower = np.nanpercentile(y_rep, 2.5, axis=0)
    upper = np.nanpercentile(y_rep, 97.5, axis=0)

    plt.figure(figsize=(10, 5))
    plt.fill_between(x[1:], lower, upper, color='lightblue', alpha=0.5, label='95% CI')
    #plt.plot(x[1:], mean_pred, label='Posterior Mean', color='blue')
    plt.scatter(x, y, color=cmapper(id), s=10, label='Observed Data')
    for _id in np.unique(id):
        mask = _id == id
        plt.plot(x[mask][1:], y[mask][1:], c=cmapper(_id))
    plt.plot(x, y, '-o', c='k')
    plt.title("Posterior Predictive Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("autoregress.png", dpi=300)

# ---- Step 4: Wrap it all up ----

def run_pipeline(stan_file_path):
    data_dict = generate_mock_data()
    fit = compile_and_run_stan(stan_file_path, data_dict)
    plot_posterior_predictive(fit, data_dict)

# ---- Entry point ----

if __name__ == "__main__":
    # Provide path to your Stan file here
    stan_file = "stan/ar.stan"
    if not os.path.exists(stan_file):
        raise FileNotFoundError(f"Stan file not found: {stan_file}")
    run_pipeline(stan_file)