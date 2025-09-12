functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N;                  // number of data points
    vector[N] r;                     // radii
    vector[N] density;               // observed log10(density)

    // OOS inputs
    int<lower=0> N_OOS;                           // number of prediction points
    vector<lower=0, upper=max(r)>[N_OOS] r_OOS;   // radii at which to predict
}

transformed data {
    vector[N] log10_density = log10(density);
    real median_r = quantile(r, 0.5);
    int N_GQ = N + N_OOS;
    vector<lower=0, upper=max(r)>[N_GQ] r_GQ = append_row(r, r_OOS);
}

parameters {
    real log10rhoS;       // log10 scale density
    real<lower=-5, upper=2> log10rS;         // log10 scale radius
    real<lower=0> a;      // inner slope transition sharpness
    real<lower=0> b;      // outer slope
    real g_raw;               // inner slope
    real<lower=0> err0;    // scatter at pivot radius
    real err_grad;        // scatter gradient
}

transformed parameters {
    array[7] real lprior;
    lprior[1] = normal_lpdf(log10rhoS | mean(log10_density), 5);
    lprior[2] = normal_lpdf(log10rS | 0.1, 1);
    lprior[3] = normal_lpdf(a | 0, 4);
    lprior[4] = normal_lpdf(b | 0, 4);
    lprior[5] = gamma_lpdf(g_raw | 3, 1.5);
    lprior[6] = normal_lpdf(err0 | 0, 1);
    lprior[7] = normal_lpdf(err_grad | 0, 1);

    real rS = pow(10., log10rS);
    real g = -(g_raw - 5);
}

model {
    target += sum(lprior);
    target += normal_lpdf(log10_density | abg_density_vec(r, log10rhoS, log10rS, a, b, g), radially_vary_err(r, err0, err_grad, median_r));
}

generated quantities {
    vector[N_GQ] log10_rho_mean;   // mean model prediction
    vector[N_GQ] log10_rho_posterior;   // posterior predictive draw (with noise)  
    vector[N_GQ] rho_posterior;
    vector[N_GQ] err_posterior = radially_vary_err(r_GQ, err0, err_grad, median_r);

    // push forward data
    log10_rho_mean = abg_density_vec(r_GQ, log10rhoS, log10rS, a, b, g);
    for(i in 1:N_GQ){
        log10_rho_posterior[i] = normal_rng(log10_rho_mean[i], err_posterior[i]);
    }

    rho_posterior = pow(10., log10_rho_posterior);
}
