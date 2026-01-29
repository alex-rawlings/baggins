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
    real rb = trunc_rayleigh_rng(0.5, 0, 1);
    real Re = trunc_rayleigh_rng(2., 0, 5);
    real n = trunc_normal_rng(4., 2., 0, 20);
    real g = trunc_normal_rng(0., 0.5, -2, 2);
    real log10rhob = trunc_normal_rng(3., 1., -5, 15);
    real a = trunc_rayleigh_rng(0.5, 0, 15);
    real err = lower_trunc_normal_rng(0., 1., 0);
    // generate data replication
    vector[N] log10_rho_prior;
    vector[N] rho_prior;

    {
        // some helper quantities
        real b = sersic_b_parameter(n);
        real p = p_parameter(n);
        real pt = terzic_preterm(g, a, rb, Re, n, b, p);

        // push forward data
        vector[N] mean_gsd = terzic_density_vec(r, pt, g, a, rb, n, b, p, Re, log10rhob);
        for(i in 1:N){
            log10_rho_prior[i] = trunc_normal_rng(mean_gsd[i], err, -5, 15);
        }
    }
    rho_prior = pow(10., log10_rho_prior);
}