functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N_obs;                  // number of data points
    vector[N_obs] r;                     // radii
    vector[N_obs] density;               // observed log10(density)

    // OOS inputs
    int<lower=0> N_OOS;                           // number of prediction points
    vector<lower=0, upper=max(r)>[N_OOS] r_OOS;   // radii at which to predict
}

transformed data {
    vector[N_obs] log10_density = log10(density);
    real median_r = quantile(r, 0.5);
    int N_GQ = N_obs + N_OOS;
    vector<lower=0, upper=max(r)>[N_GQ] r_GQ = append_row(r, r_OOS);
}

parameters {
    real<lower=-5, upper=10> log10rhoS;       // log10 scale density
    real<lower=-5, upper=2> log10rS;         // log10 scale radius
    real log10a;      // inner slope transition sharpness
    real b;      // outer slope
    real<lower=0> g;
    real<lower=0> err; // observation scatter
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
    vector[N_GQ] log10_rho_mean;   // mean model prediction
    vector[N_GQ] log10_rho_posterior;   // posterior predictive draw (with noise)  
    vector[N_GQ] rho_posterior;

    // push forward data
    log10_rho_mean = abg_density_vec(r_GQ, log10rhoS, log10rS, log10a, b, g);
    for(i in 1:N_GQ){
        log10_rho_posterior[i] = normal_rng(log10_rho_mean[i], err);
    }

    rho_posterior = pow(10., log10_rho_posterior);
}
