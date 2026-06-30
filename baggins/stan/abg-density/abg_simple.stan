functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N_obs;                  // number of data points
    vector[N_obs] r;                     // radii
    vector[N_obs] density;               // density

    // OOS inputs
    int<lower=0> N_OOS;                           // number of prediction points
    vector<lower=0>[N_OOS] r_OOS;   // radii at which to predict
}

transformed data {
    vector[N_obs] log10_density = log10(density);
}

parameters {
    real<lower=-5, upper=10> log10rhoS;       // log10 scale density
    real<lower=-5, upper=2> log10rS;          // log10 scale radius
    real log10a;                              // transition sharpness
    real b;                                   // outer slope
    real<lower=0> g;                          // inner slope
    real<lower=0> err;                        // observation scatter
}

transformed parameters {
    array[6] real lprior;
    lprior[1] = normal_lpdf(log10rhoS | 5, 3);
    lprior[2] = normal_lpdf(log10rS | 0.1, 1);
    lprior[3] = normal_lpdf(log10a | 0.5, 0.5);
    lprior[4] = normal_lpdf(b | 0, 4);
    lprior[5] = normal_lpdf(g | 0, 3);
    lprior[6] = normal_lpdf(err| 0, 1);
}

model {
    target += sum(lprior);
    target += normal_lpdf(log10_density | abg_density_vec(r, log10rhoS, log10rS, log10a, b, g), err);
}

generated quantities {
    real rS = pow(10., log10rS);
    real a = pow(10, log10a);

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
    log10_rho_mean = abg_density_vec(r, log10rhoS, log10rS, log10a, b, g);
    for (i in 1:N_obs) {
        log10_density_posterior[i] = normal_rng(log10_rho_mean[i], err);
        log_lik[i] = normal_lpdf(log10_density[i] | log10_rho_mean[i], err);
    }
    density_posterior = pow(10., log10_density_posterior);

    // OOS
    log10_rho_mean_OOS = abg_density_vec(r_OOS, log10rhoS, log10rS, log10a, b, g);
    for (i in 1:N_OOS) {
        log10_density_OOS[i] = normal_rng(log10_rho_mean_OOS[i], err);
    }
    density_OOS = pow(10., log10_density_OOS);
}
