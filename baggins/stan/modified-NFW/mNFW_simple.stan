functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N_obs;                  // number of data points
    vector[N_obs] r;                     // radii
    vector[N_obs] log10_density;               // observed log10(density)

    // OOS inputs
    int<lower=0> N_OOS;                           // number of prediction points
    vector<lower=0, upper=max(r)>[N_OOS] r_OOS;   // radii at which to predict
}

parameters {
    real<lower=0, upper=15> log10rhoS;       // log10 scale density
    real<lower=-2, upper=2.7> log10rS;         // log10 scale radius, extends to ~500kpc
    real<lower=-2, upper=1> log10g;
    real<lower=0> err; // observation scatter
}

transformed parameters {
    array[4] real lprior;
    lprior[1] = normal_lpdf(log10rhoS | 10, 3);
    lprior[2] = normal_lpdf(log10rS | 1, 2);
    lprior[3] = normal_lpdf(log10g | 0, 0.25);
    lprior[4] = normal_lpdf(err| 0, 1);
}

model {
    target += sum(lprior);
    target += normal_lpdf(log10_density | mNFW_density_vec(r, log10rhoS, log10rS, log10g), err);
}


generated quantities {
    real rS = pow(10., log10rS);
    real g = pow(10., log10g);

    // In-sample posterior predictive
    vector[N_obs] log10_rho_mean;
    vector[N_obs] log10_density_posterior;
    vector[N_obs] density_posterior;
    vector[N_obs] log_lik;

    // Out-of-sample predictions
    vector[N_OOS] log10_rho_mean_OOS;
    vector[N_OOS] log10_density_OOS;
    vector[N_OOS] density_OOS;

    // In-sample
    log10_rho_mean = mNFW_density_vec(r, log10rhoS, log10rS, log10g);
    for (i in 1:N_obs) {
        log10_density_posterior[i] = normal_rng(log10_rho_mean[i], err);
        log_lik[i] = normal_lpdf(log10_density[i] | log10_rho_mean[i], err);
    }
    density_posterior = pow(10., log10_density_posterior);

    // OOS
    log10_rho_mean_OOS = mNFW_density_vec(r_OOS, log10rhoS, log10rS, log10g);
    for (i in 1:N_OOS) {
        log10_density_OOS[i] = normal_rng(log10_rho_mean_OOS[i], err);
    }
    density_OOS = pow(10., log10_density_OOS);
}
