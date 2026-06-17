functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N;                  // number of data points
    vector[N] r;                     // radii
    vector[N] density;               // observed log10(density)
}

transformed data {
    real median_r = quantile(r, 0.5);
    real mean_log10_dens = mean(log10(density));
}

generated quantities {
    // prior distributions
    real log10rhoS = trunc_normal_rng(mean_log10_dens, 2, 0, 15);
    real log10rS = trunc_normal_rng(0.1, 1, -5, 2);
    real log10g = trunc_normal_rng(0, 0.5, -2, 2);
    real err = lower_trunc_normal_rng(0, 1, 0);

    // transformed parameters
    real rS = pow(10., log10rS);
    real g = pow(10., log10g);

    vector[N] log10_rho_mean;   // mean model prediction
    vector[N] log10_rho_prior;   // posterior predictive draw (with noise)  
    vector[N] rho_prior;

    // push forward data
    log10_rho_mean = mNFW_density_vec(r, log10rhoS, log10rS, log10g);
    for(i in 1:N){
        log10_rho_prior[i] = normal_rng(log10_rho_mean[i], err[i]);
    }

    rho_prior = pow(10., log10_rho_prior);
}
