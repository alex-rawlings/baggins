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
    real log10rhoS = normal_rng(mean_log10_dens, 2);
    real log10rS = trunc_normal_rng(0.1, 1, -5, 2);
    real a = lower_trunc_normal_rng(0, 4, 0);
    real b = lower_trunc_normal_rng(0, 4, 0);
    real g_raw = gamma_rng(3, 1.5);
    real err0 = lower_trunc_normal_rng(0, 1, 0);
    real err_grad = normal_rng(0, 1);

    // transformed parameters
    real rS = pow(10., log10rS);
    real g = -(g_raw - 5);

    vector[N] log10_rho_mean;   // mean model prediction
    vector[N] log10_rho_prior;   // posterior predictive draw (with noise)  
    vector[N] rho_prior;
    vector[N] err_prior = radially_vary_err(r, err0, err_grad, median_r);

    // push forward data
    log10_rho_mean = abg_density_vec(r, log10rhoS, log10rS, a, b, g);
    for(i in 1:N){
        log10_rho_prior[i] = normal_rng(log10_rho_mean[i], err_prior[i]);
    }

    rho_prior = pow(10., log10_rho_prior);
}
