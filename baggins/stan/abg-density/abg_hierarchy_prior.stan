functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N_obs;
    int<lower=1> N_group;
    array[N_obs] int<lower=1, upper=N_group> group_id;
    vector<lower=0>[N_obs] r;
}


generated quantities {
    // --- Hyperparameters drawn from priors ---
    real log10rhoS_mean = normal_rng(5, 1);
    real log10rhoS_std = lower_trunc_normal_rng(0, 1, 0);
    real log10rS_mean = normal_rng(0, 1);
    real log10rS_std = lower_trunc_normal_rng(0, 0.5, 0);
    real a_mean = lower_trunc_normal_rng(0, 4, 0);
    real a_std = lower_trunc_normal_rng(0, 2, 0);
    real b_mean = normal_rng(0, 4);
    real b_std = lower_trunc_normal_rng(0, 2, 0);
    real g_mean = normal_rng(0, 2);
    real g_std = normal_rng(0, 2);
    real err0 = lower_trunc_normal_rng(0, 1, 0);
    real err_grad = normal_rng(0, 1);

    cholesky_factor_corr[5] L_corr = lkj_corr_cholesky_rng(5, 2.0);
    matrix[5, 5] L = diag_pre_multiply([log10rhoS_std, log10rS_std, a_std, b_std, g_std], L_corr);

    real obs_sigma = lower_trunc_normal_rng(0, 1, 0);

    // define latent parameters for each group
    vector[N_group] log10rS;
    vector[N_group] log10rhoS;
    vector[N_group] a;
    vector[N_group] b;
    vector[N_group] g;

    // --- Group-level parameters from MVN prior ---
    array[N_group] vector[5] theta_group;
    for (s in 1:N_group) {
        theta_group[s] = multi_normal_cholesky_rng([log10rhoS_mean, log10rS_mean, a_mean, b_mean, g_mean]', L);
        log10rhoS[s] = theta_group[s][1];
        log10rS[s] = theta_group[s][2];
        a[s] = theta_group[s][3];
        b[s] = theta_group[s][4];
        g[s] = theta_group[s][5];
    }

    // --- Prior predictive for observed radii ---
    vector[N_obs] log10_rho_prior;
    vector[N_obs] rho_prior;
    vector[N_obs] log10_rho_mean;

    log10_rho_mean = abg_density_vec(r, log10rhoS[group_id], log10rS[group_id], a[group_id], b[group_id], g[group_id]);

    for (i in 1:N_obs) {
        log10_rho_prior[i] = trunc_normal_rng(log10_rho_mean[i], obs_sigma, -5, 15);
    }
    rho_prior = pow(10., log10_rho_prior);
}
