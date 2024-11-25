functions{
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    vector<lower=0.0>[N_tot] R;
}


generated quantities {
    // latent quantities
    real rb = trunc_rayleigh_rng(1, 0.1, 3);
    real Re = trunc_rayleigh_rng(8, 0.1, 20);
    real n = trunc_normal_rng(4, 1, 0, 15);
    real g = trunc_exponential_rng(2, 0, 1);
    real log10densb = trunc_normal_rng(10, 1, 5, 15);
    real a = trunc_rayleigh_rng(7, 0, 15);
    vector[N_tot] err;

    // prior check
    vector[N_tot] log10_surf_rho_prior;
    vector[N_tot] surf_rho_prior;


    {
        // helper quantities
        real b_param = sersic_b_parameter(n);
        real preterm = graham_preterm(
            g,
            a,
            n,
            b_param,
            rb,
            Re
            );

        // push forward data
        vector[N_tot] mean_gsd = graham_surf_density_vec(
            R,
            preterm,
            g,
            a,
            rb,
            n,
            b_param,
            Re,
            log10densb
        );
        for(i in 1:N_tot){
            err[i] = lower_trunc_normal_rng(0, 1, 0);
            log10_surf_rho_prior[i] = trunc_normal_rng(mean_gsd[i], err[i], -5, 15);
        }
    }
    surf_rho_prior = pow(10., log10_surf_rho_prior);
}