functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N_obs;                  // number of data points
    vector[N_obs] r;                     // radii
    vector[N_obs] density;               // observed log10(density)
}

transformed data {
    vector[N_obs] log10_density = log10(density);
}

generated quantities {
    // prior distributions
    real log10rhoS = trunc_normal_rng(10, 3, 0, 15);
    real log10rS = trunc_normal_rng(1, 2, -2, 2.7);
    real log10g = trunc_normal_rng(0, 0.25, -2, 1);
    real err = lower_trunc_normal_rng(0, 1, 0);

    // transformed parameters
    real rS = pow(10., log10rS);
    real g = pow(10., log10g);

    vector[N_obs] log10_rho_mean;   // mean model prediction
    vector[N_obs] log10_density_prior;   // posterior predictive draw (with noise)  
    vector[N_obs] density_prior;

    // push forward data
    log10_rho_mean = mNFW_density_vec(r, log10rhoS, log10rS, log10g);
    for(i in 1:N_obs){
        log10_density_prior[i] = normal_rng(log10_rho_mean[i], err);
    }

    density_prior = pow(10., log10_density_prior);
}
