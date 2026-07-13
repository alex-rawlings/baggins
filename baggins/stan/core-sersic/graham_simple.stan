functions{
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N_obs;                // number of data points
    vector<lower=0.001>[N_obs] R;      // radii
    array[N_obs] real log10_density;  // density

    // Out of Sample points
    int<lower=1> N_OOS;                           // number of prediction points
    vector<lower=min(R), upper=max(R)>[N_OOS] R_OOS;  // radii at which to predict
}


transformed data {
    vector[N_obs] log10_density = log10(density);
}


parameters {
    // no hierarchy: assume all observations from the same set
    real<lower=0, upper=5> rb;
    real<lower=0, upper=20> Re;
    real<lower=-5, upper=15> log10densb;
    real<lower=0, upper=1> g;
    real<lower=0, upper=20> n;
    real<lower=0, upper=15> a;

    // model variance, function of radius
    real<lower=0> err;
}


transformed parameters {
    array[7] real lprior;
    lprior[1] = rayleigh_lpdf(rb | 1.);
    lprior[2] = rayleigh_lpdf(Re | 8.);
    lprior[3] = normal_lpdf(n | 4., 1.);
    lprior[4] = exponential_lpdf(g | 2.);
    lprior[5] = normal_lpdf(log10densb | 10., 1.);
    lprior[6] = rayleigh_lpdf(a | 7.);
    lprior[7] = normal_lpdf(err | 0., 1.);
}


model{
    // density at model parameters
    target += sum(lprior);

    // likelihood
    {
        real b = sersic_b_parameter(n);
        real pt = graham_preterm(g, a, n, b, rb, Re);
        target += normal_lpdf(log10_density | graham_surf_density_vec(R, pt, g, a, rb, n, b, Re, log10densb), err);
    }
}


generated quantities {
    // In-sample posterior predictive
    vector[N_obs] log10_Sigma_mean;
    vector[N_obs] log10_density_posterior;
    vector[N_obs] density_posterior;
    vector[N_obs] log_lik;

    // Out-of-sample predictions
    vector[N_OOS] log10_Sigma_mean_OOS;
    vector[N_OOS] log10_density_OOS;
    vector[N_OOS] density_OOS;

    {
        // In-sample
        // some helper quantities
        real b = sersic_b_parameter(n);
        real pt = graham_preterm(g, a, n, b, rb, Re);

        log10_Sigma_mean = graham_surf_density_vec(R, pt, g, a, rb, n, b, Re, log10densb);
        for (i in 1:N_obs){
            log10_density_posterior[i] = trunc_normal_rng(log10_Sigma_mean[i], err, -5, 15);
            log_lik[i] = normal_lpdf(log10_density[i] | log10_Sigma_mean[i], err);
        }

        // OOS
        log10_Sigma_mean_OOS = graham_surf_density_vec(R_OOS, pt, g, a, rb, n, b, Re, log10densb);
        for(i in 1:N_OOS){
            log10_density_OOS[i] = trunc_normal_rng(log10_Sigma_mean_OOS[i], err, -5, 15);
        }
    }
    density_OOS = pow(10., log10_density_OOS);
}
