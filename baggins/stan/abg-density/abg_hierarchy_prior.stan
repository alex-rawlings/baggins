functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    vector<lower=0>[N_tot] R;

    // Individual groups
    // number of groups
    int<lower=1> N_groups;
    // indexing of observations to group
    array[N_tot] int<lower=1, upper=N_groups> group_idx;
}


generated quantities {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real log10rhoS_mean = normal_rng(4, 2);
    real log10rhoS_std = lower_trunc_normal_rng(0, 1, 0);
    real rS_sig = lower_trunc_normal_rng(0, 3, 0);
    real a_mean = normal_rng(0, 4);
    real a_std = lower_trunc_normal_rng(0, 2, 0);
    real b_mean = normal_rng(0, 4);
    real b_std = lower_trunc_normal_rng(0, 2, 0);
    real g_mean = normal_rng(0, 4);
    real g_std = lower_trunc_normal_rng(0, 2, 0);
    real err = lower_trunc_normal_rng(0, 4, 0);

    // define latent parameters for each group
    vector[N_groups] rS;
    vector[N_groups] log10rhoS;
    vector[N_groups] a;
    vector[N_groups] b;
    vector[N_groups] g;


    // prior check
    vector[N_tot] log10_rho_prior;
    vector[N_tot] rho_prior;

    // sample latent parameters and prior check
    {
        for(i in 1:N_groups){
            log10rhoS[i] = trunc_normal_rng(log10rhoS_mean, log10rhoS_std, -5, 15);
            rS[i] = trunc_rayleigh_rng(rS_sig, 0, 5);
            a[i] = normal_rng(a_mean, a_std);
            b[i] = normal_rng(b_mean, b_std);
            g[i] = normal_rng(g_mean, g_std);
        }

        // push forward data
        vector[N_tot] mean_dens = abg_density_vec(
                                    r,
                                    log10rhoS[group_idx],
                                    rS[group_idx],
                                    a[group_idx],
                                    b[group_idx],
                                    g[group_idx]
                                )
        for(i in 1:N_tot){
            log10_surf_rho_prior[i] = trunc_normal_rng(mean_dens[i], err, -5, 15);
        }
    }
    surf_rho_prior = pow(10., log10_surf_rho_prior);
}