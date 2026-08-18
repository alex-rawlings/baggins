functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}

data {
    int<lower=1> N_obs;                  // number of data points
    vector[N_obs] r;                     // radii
    vector[N_obs] density;               // observed density

    real<lower=0> M_BH;                  // central black hole mass [Msun]

    // OOS inputs
    int<lower=0> N_OOS;                           // number of prediction points
    vector<lower=0>[N_OOS] r_OOS;   // radii at which to predict
}

transformed data {
    vector[N_obs] log10_density = log10(density);
    real Gconst = 4.30091e-6;    // kpc (km/s)^2 / Msun
    real c_light = 299792.458;   // km/s
    // profile is cut off at twice the BH Schwarzschild radius, following
    // Alonso-Alvarez, Cline & Dewar (2024), arXiv:2401.14450
    real r_min = 4. * Gconst * M_BH / square(c_light);   // kpc
}

parameters {
    real<lower=0, upper=15> log10rhoS;       // log10 scale density
    real<lower=-2, upper=2.7> log10rS;         // log10 scale radius, extends to ~500kpc
    real<lower=-2, upper=1> log10g;
    real<lower=0.5, upper=7./3.> gamma_sp;     // DM spike slope: 1/2 (post-merger) to 7/3 (adiabatic growth)
    real<lower=0> err; // observation scatter
}

transformed parameters {
    array[5] real lprior;
    lprior[1] = normal_lpdf(log10rhoS | 10, 3);
    lprior[2] = normal_lpdf(log10rS | 1, 2);
    lprior[3] = normal_lpdf(log10g | 0, 0.25);
    lprior[4] = normal_lpdf(gamma_sp | 1.4, 0.5);
    lprior[5] = normal_lpdf(err| 0, 1);
}

model {
    target += sum(lprior);
    target += normal_lpdf(log10_density | mNFWspike_density_vec(r, log10rhoS, log10rS, log10g, M_BH, gamma_sp, r_min), err);
}


generated quantities {
    real rS = pow(10., log10rS);
    real g = pow(10., log10g);
    real rhoS = pow(10., log10rhoS);
    real r_sp = 0.2 * sqrt(M_BH / (pi() * rhoS * rS));   // DM spike radius

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
    log10_rho_mean = mNFWspike_density_vec(r, log10rhoS, log10rS, log10g, M_BH, gamma_sp, r_min);
    for (i in 1:N_obs) {
        log10_density_posterior[i] = normal_rng(log10_rho_mean[i], err);
        log_lik[i] = normal_lpdf(log10_density[i] | log10_rho_mean[i], err);
    }
    density_posterior = pow(10., log10_density_posterior);

    // OOS
    log10_rho_mean_OOS = mNFWspike_density_vec(r_OOS, log10rhoS, log10rS, log10g, M_BH, gamma_sp, r_min);
    for (i in 1:N_OOS) {
        log10_density_OOS[i] = normal_rng(log10_rho_mean_OOS[i], err);
    }
    density_OOS = pow(10., log10_density_OOS);
}
