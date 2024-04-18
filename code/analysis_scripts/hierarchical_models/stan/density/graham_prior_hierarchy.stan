functions {
    #include funcs_graham.stan
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
    real log10densb_mean = normal_rng(10, 2);
    real log10densb_std = lower_trunc_normal_rng(0, 1, 0);
    real g_lam = exponential_rng(10);
    real rb_sig = lower_trunc_normal_rng(0, 1, 0);
    real n_mean = trunc_normal_rng(8, 4, 0, 15);
    real n_std = lower_trunc_normal_rng(0, 4, 0);
    real a_sig = gamma_rng(2, 0.2);
    real Re_sig = lower_trunc_normal_rng(0, 12, 0);
    real err_mean = lower_trunc_normal_rng(0, 1, 0);
    real err_std = lower_trunc_normal_rng(0, 0.2, 0);

    // model variance, function of radius
    vector<lower=0>[N_tot] err;

    // define latent parameters for each group
    vector[N_groups] rb;
    vector[N_groups] Re;
    vector[N_groups] n;
    vector[N_groups] g;
    vector[N_groups] log10densb;
    vector[N_groups] a;


    // prior check
    vector[N_tot] log10_surf_rho_prior;
    vector[N_tot] surf_rho_prior;

    // sample latent parameters and prior check
    {
        vector[N_groups] b_param;
        vector[N_groups] pre_term;

        for(i in 1:N_groups){
            log10densb[i] = trunc_normal_rng(log10densb_mean, log10densb_std, -5, 15);
            g[i] = trunc_exponential_rng(g_lam, 0, 2);
            rb[i] = trunc_rayleigh_rng(rb_sig, 0, 5);
            n[i] = trunc_normal_rng(n_mean, n_std, 0, 20);
            a[i] = trunc_rayleigh_rng(a_sig, 0, 15);
            Re[i] = trunc_rayleigh_rng(Re_sig, 0, 20);
        }

        // some helper quantities
        b_param = sersic_b_parameter(n);
        pre_term = graham_preterm(g, a, n, b_param, rb, Re);

        // push forward data
        vector[N_tot] mean_gsd = graham_surf_density_vec(
                                        R,
                                        pre_term[group_idx],
                                        g[group_idx],
                                        a[group_idx],
                                        rb[group_idx],
                                        n[group_idx],
                                        b_param[group_idx],
                                        Re[group_idx],
                                        log10densb[group_idx]);
        for(i in 1:N_tot){
            err[i] = lower_trunc_normal_rng(err_mean, err_std, 0.);
            log10_surf_rho_prior[i] = trunc_normal_rng(mean_gsd[i], err[i], -5, 15);
        }
    }
    surf_rho_prior = pow(10., log10_surf_rho_prior);
}