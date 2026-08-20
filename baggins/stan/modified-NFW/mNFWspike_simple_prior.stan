functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N_obs;                  // number of data points
    vector[N_obs] r;                     // radii
    vector[N_obs] density;               // observed log10(density)

    real<lower=0> M_BH;                  // central black hole mass [Msun]
}

transformed data {
    vector[N_obs] log10_density = log10(density);
    real Gconst = 4.30091e-6;    // kpc (km/s)^2 / Msun
    real c_light = 299792.458;   // km/s
    real r_min = 4. * Gconst * M_BH / square(c_light);   // kpc, twice the Schwarzschild radius
}

generated quantities {
    // prior distributions
    real log10rhoS = trunc_normal_rng(10, 3, 0, 15);
    real log10rS = trunc_normal_rng(1, 2, -2, 2.7);
    real log10g = trunc_normal_rng(0, 0.25, -2, 1);
    real gamma_sp = trunc_normal_rng(1.4, 0.5, 0.5, 7./3.);
    real err = lower_trunc_normal_rng(0, 1, 0);

    // transformed parameters
    real rS = pow(10., log10rS);
    real g = pow(10., log10g);
    real rhoS = pow(10., log10rhoS);
    real r_sp = 0.2 * sqrt(M_BH / (pi() * rhoS * rS));   // DM spike radius

    vector[N_obs] log10_rho_mean;   // mean model prediction
    vector[N_obs] log10_density_prior;   // posterior predictive draw (with noise)
    vector[N_obs] density_prior;

    // push forward data
    log10_rho_mean = mNFWspike_density_vec(r, log10rhoS, log10rS, log10g, M_BH, gamma_sp, r_min);
    for(i in 1:N_obs){
        log10_density_prior[i] = normal_rng(log10_rho_mean[i], err);
    }

    density_prior = pow(10., log10_density_prior);
}
