functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N;
    vector<lower=0>[N] r;
    vector<lower=0>[N] density;
}


transformed data {
    vector[N] log10_density = log10(density);
}


generated quantities {
    // latent quantities
    real rS = trunc_rayleigh_rng(0.5, 0, 2);
    real a = lower_trunc_normal_rng(0, 4, 0);
    real b = lower_trunc_normal_rng(0, 4, 0);
    real g = lower_trunc_normal_rng(0, 4, 0);
    real log10rhoS = trunc_normal_rng(3., 1., -5, 15);
    real err = lower_trunc_normal_rng(0., 1., 0);

    // generate data replication
    vector[N] log10_rho_prior;
    vector[N] rho_prior;

    log10_rho_prior = to_vector(normal_rng(abg_density_vec(r, log10rhoS, rS, a, b, g), err));
    rho_prior = pow(10., log10_rho_prior);
}