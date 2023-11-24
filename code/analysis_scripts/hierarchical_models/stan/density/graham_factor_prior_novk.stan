functions {
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    // Observations
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    array[N_tot] real<lower=0.0> R;

    // Individual Contexts
    // total number of individual contexts
    int<lower=1> N_contexts;
    // indexing of observations to context level
    array[N_tot] int<lower=1, upper=N_contexts> context_idx;

    // Factor levels
    // number of factors
    int<lower=1> N_factors;
    // indexing of contexts to factor level
    array[N_contexts] int<lower=1> factor_idx;
}


generated quantities {
    // hyperpriors
    real log10densb_mean_hyp1 = normal_rng(10, 0.5);
    real log10densb_mean_hyp2 = lower_trunc_normal_rng(0, 0.05, 0);
    real log10densb_std_hyp = lower_trunc_normal_rng(0, 0.05, 0);
    real g_hyp = exponential_rng(10);
    real rb_hyp = rayleigh_rng(0.1);
    real n_mean_hyp1 = trunc_normal_rng(4, 2, 0, 20);
    real n_mean_hyp2 = lower_trunc_normal_rng(0, 0.1, 0);
    real n_std_hyp1 = exponential_rng(0.1);
    real n_std_hyp2 = lower_trunc_normal_rng(0, 0.1, 0);
    real a_hyp = rayleigh_rng(10);
    real Re_hyp = rayleigh_rng(10);
    real err_hyp = lower_trunc_normal_rng(0, 0.01, 0);

    // parameters for each factor level
    vector[N_factors] log10densb_mean;
    vector[N_factors] log10densb_std;
    vector[N_factors] g_lam;
    vector[N_factors] rb_sig;
    vector[N_factors] n_mean;
    vector[N_factors] n_std;
    vector[N_factors] a_sig;
    vector[N_factors] Re_sig;
    vector[N_factors] err;

    // parameters for each context
    vector[N_contexts] rb;
    vector[N_contexts] Re;
    vector[N_contexts] n;
    vector[N_contexts] g;
    vector[N_contexts] log10densb;
    vector[N_contexts] a;

    // push forward data array
    array[N_tot] real log10_surf_rho_prior;

    // factor level quantities
    for(i in 1:N_factors){
        log10densb_mean[i] = normal_rng(log10densb_mean_hyp1, log10densb_mean_hyp2);
        log10densb_std[i] = lower_trunc_normal_rng(0, log10densb_std_hyp, 0);
        g_lam[i] = rayleigh_rng(g_hyp);
        rb_sig[i] = rayleigh_rng(rb_hyp);
        n_mean[i] = trunc_normal_rng(n_mean_hyp1, n_mean_hyp2, 0, 15);
        n_std[i] = trunc_normal_rng(0, n_std_hyp2, 0, 15);
        a_sig[i] = rayleigh_rng(a_hyp);
        Re_sig[i] = rayleigh_rng(Re_hyp);
        err[i] = lower_trunc_normal_rng(0, err_hyp, 0);
    }

    // context level quantities
    {
        vector[N_contexts] pre_term;
        vector[N_contexts] b_param;
        for(i in 1:N_contexts){
            log10densb[i] = trunc_normal_rng(log10densb_mean[factor_idx[i]], log10densb_std[factor_idx[i]], -5, 15);
            g[i] = trunc_exponential_rng(g_lam[factor_idx[i]], 0, 2);
            rb[i] = trunc_rayleigh_rng(rb_sig[factor_idx[i]], 0, 5);
            n[i] = trunc_normal_rng(n_mean[factor_idx[i]], n_std[factor_idx[i]], 0, 20);
            a[i] = rayleigh_rng(a_sig[factor_idx[i]]);
            Re[i] = trunc_rayleigh_rng(Re_sig[factor_idx[i]], 0, 20);
            // some helper quantities
            b_param = sersic_b_parameter(n);
            pre_term = graham_preterm(g, a, n, b_param, rb, Re);
        }

        // push forward data
        real mean_gsd;
        for(i in 1:N_tot){
            mean_gsd = graham_surf_density(
                            R[i],
                            pre_term[context_idx[i]],
                            g[context_idx[i]],
                            a[context_idx[i]],
                            rb[context_idx[i]],
                            n[context_idx[i]],
                            b_param[context_idx[i]],
                            Re[context_idx[i]],
                            log10densb[context_idx[i]]
                        );
            log10_surf_rho_prior[i] = trunc_normal_rng(mean_gsd, err[factor_idx[context_idx[i]]], -5, 15);
        }
    }
}