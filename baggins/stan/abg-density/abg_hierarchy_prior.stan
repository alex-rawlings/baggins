functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    // total number of points
    int<lower=1> N;
    // array of radial values
    vector<lower=0>[N] r;

    // Individual groups
    // number of groups
    int<lower=1> N_groups;
    // indexing of observations to group
    array[N] int<lower=1, upper=N_groups> group_idx;
}

transformed data {
    real median_r = quantile(r, 0.5);
}

generated quantities {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real log10rhoS_mean = normal_rng(5, 1);
    real log10rhoS_std = lower_trunc_normal_rng(0, 1, 0);
    real log10rS_mean = normal_rng(0, 1);
    real log10rS_std = lower_trunc_normal_rng(0, 0.5, 0);
    real a_mean = normal_rng(0, 4);
    real a_std = lower_trunc_normal_rng(0, 2, 0);
    real b_mean = normal_rng(0, 4);
    real b_std = lower_trunc_normal_rng(0, 2, 0);
    real g_raw_a = uniform_rng(-3, 0);
    real g_raw_b = uniform_rng(0, 3);
    real err0 = lower_trunc_normal_rng(0, 1, 0);
    real err_grad = normal_rng(0, 1);

    // define latent parameters for each group
    vector[N_groups] log10rS;
    vector[N_groups] log10rhoS;
    vector[N_groups] a;
    vector[N_groups] b;
    vector[N_groups] g;

    vector[N_groups] rS;
    vector[N_groups] g_raw;


    // prior check
    vector[N] log10_rho_prior;
    vector[N] log10_rho_mean;
    vector[N] rho_prior;

    vector[N] err_prior = radially_vary_err(r, err0, err_grad, median_r);

    // sample latent parameters and prior check
    {
        for(i in 1:N_groups){
            log10rhoS[i] = trunc_normal_rng(log10rhoS_mean, log10rhoS_std, -5, 15);
            rS[i] = pow(10., log10rS[i]);
            log10rS[i] = trunc_normal_rng(log10rS_mean, log10rS_std, -5, 2);
            a[i] = normal_rng(a_mean, a_std);
            b[i] = normal_rng(b_mean, b_std);
            g_raw[i] = uniform_rng(g_raw_a, g_raw_b);
            g[i] = g_raw[i];
        }

        // push forward data
        log10_rho_mean = abg_density_vec(
                                    r,
                                    log10rhoS[group_idx],
                                    log10rS[group_idx],
                                    a[group_idx],
                                    b[group_idx],
                                    g[group_idx]
                                );
        for(i in 1:N){
            log10_rho_prior[i] = trunc_normal_rng(log10_rho_mean[i], err_prior[i], -5, 15);
        }
    }
    rho_prior = pow(10., log10_rho_prior);
}