functions {
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    // Observations
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    vector<lower=0>[N_tot] R;
    // array of surface density values
    array[N_tot] real log10_surf_rho;

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

    // Out of Sample points
    // follows same structure as above
    // total number of OOS points
    int<lower=1> N_OOS;
    // total number of individual OOS contexts
    int<lower=1> N_contexts_OOS;
    // total number of individual OOS factors
    int<lower=1> N_factors_OOS;
    // OOS radii values
    vector<lower=0, upper=max(R)>[N_OOS] R_OOS;

    // OOS contexts
    // total number of individual OOS contexts
    // context ids for generated quantities
    array[N_OOS] int<lower=1> context_idx_OOS;

    // OOS factors
    // total number of individual OOS factors
    //
    // factor ids for generated quantities
    array[N_contexts_OOS] int<lower=1> factor_idx_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
    int N_contexts_GQ = N_contexts + N_contexts_OOS;
    int N_factors_GQ = N_factors + N_factors_OOS;
    vector<lower=0, upper=max(R)>[N_GQ] R_GQ = append_row(R, R_OOS);
    array[N_GQ] int<lower=1> context_idx_GQ = append_array(context_idx, context_idx_OOS);
    array[N_contexts_GQ] int<lower=1> factor_idx_GQ = append_array(factor_idx, factor_idx_OOS);
}


parameters {
    // hyperpriors
    real log10densb_mean_hyp1;
    real<lower=0> log10densb_mean_hyp2;
    real<lower=0> log10densb_std_hyp;
    real<lower=0> g_hyp;
    real<lower=0> rb_hyp;
    real<lower=0, upper=20> n_mean_hyp1;
    real<lower=0> n_mean_hyp2;
    real<lower=0> n_std_hyp1;
    real<lower=0> n_std_hyp2;
    real<lower=0> a_hyp;
    real<lower=0> Re_hyp;
    real<lower=0> err_hyp;

    // parameters for each factor level
    vector[N_factors] log10densb_mean;
    vector<lower=0>[N_factors] log10densb_std;
    vector<lower=0>[N_factors] g_lam;
    vector<lower=0>[N_factors] rb_sig;
    vector<lower=0, upper=15>[N_factors] n_mean;
    vector<lower=0, upper=15>[N_factors] n_std;
    vector<lower=0>[N_factors] a_sig;
    vector<lower=0>[N_factors] Re_sig;
    vector<lower=0>[N_factors] err;

    // latent parameters for each context
    vector<lower=0, upper=5>[N_contexts] rb;
    vector<lower=0, upper=20>[N_contexts] Re;
    vector<lower=0, upper=20>[N_contexts] n;
    vector<lower=0, upper=2>[N_contexts] g;
    vector<lower=-5, upper=15>[N_contexts] log10densb;
    vector<lower=0>[N_contexts] a;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[12] real lprior;
    lprior[1] = normal_lpdf(log10densb_mean_hyp1 | 10, 0.5);
    lprior[2] = normal_lpdf(log10densb_mean_hyp2 | 0, 0.05);
    lprior[3] = normal_lpdf(log10densb_std_hyp | 0, 0.05);
    lprior[4] = exponential_lpdf(g_hyp | 10);
    lprior[5] = rayleigh_lpdf(rb_hyp | 0.1);
    lprior[6] = normal_lpdf(n_mean_hyp1 | 4, 2);
    lprior[7] = normal_lpdf(n_mean_hyp2 | 0, 0.1);
    lprior[8] = exponential_lpdf(n_std_hyp1 | 0.1);
    lprior[9] = normal_lpdf(n_std_hyp2 | 0, 0.1);
    lprior[10] = rayleigh_lpdf(a_hyp | 10);
    lprior[11] = rayleigh_lpdf(Re_hyp | 10);
    lprior[12] = normal_lpdf(err_hyp | 0, 0.01);
}


model {
    // density at hyperparameters
    target += sum(lprior);

    // connect hyperparameters to factor parameters
    // for each factor
    target += normal_lpdf(log10densb_mean | log10densb_mean_hyp1, log10densb_mean_hyp2);
    target += normal_lpdf(log10densb_std | 0, log10densb_std_hyp);
    target += rayleigh_lpdf(g_lam | g_hyp);
    target += rayleigh_lpdf(rb_sig | rb_hyp);
    target += normal_lpdf(n_mean | n_mean_hyp1, n_mean_hyp2);
    target += normal_lpdf(n_std | n_std_hyp1, n_std_hyp2);
    target += rayleigh_lpdf(a_sig | a_hyp);
    target += rayleigh_lpdf(Re_sig | Re_hyp);
    target += normal_lpdf(err | 0, err_hyp);

    // connect factor parameters to context parameters
    // for each context
    target += normal_lpdf(log10densb | log10densb_mean[factor_idx], log10densb_std[factor_idx]);
    target += exponential_lpdf(g | g_lam[factor_idx]);
    target += rayleigh_lpdf(rb | rb_sig[factor_idx]);
    target += normal_lpdf(n | n_mean[factor_idx], n_std[factor_idx]);
    target += rayleigh_lpdf(a | a_sig[factor_idx]);
    target += rayleigh_lpdf(Re | Re_sig[factor_idx]);

    //target += normal_lpdf(log10_surf_rho | log10_surf_rho_calc, err[factor_idx[context_idx]]);
    target += reduce_sum(partial_sum, log10_surf_rho, 1, N_contexts, R, g, a, rb, n, Re, log10densb, err, factor_idx, context_idx);
}


generated quantities {
    // parameters for each factor level
    vector[N_factors_GQ] log10densb_mean_posterior;
    vector[N_factors_GQ] log10densb_std_posterior;
    vector[N_factors_GQ] g_lam_posterior;
    vector[N_factors_GQ] rb_sig_posterior;
    vector[N_factors_GQ] n_mean_posterior;
    vector[N_factors_GQ] n_std_posterior;
    vector[N_factors_GQ] a_sig_posterior;
    vector[N_factors_GQ] Re_sig_posterior;
    vector[N_factors_GQ] err_posterior;

    // parameters for each context
    vector[N_contexts_GQ] rb_posterior;
    vector[N_contexts_GQ] Re_posterior;
    vector[N_contexts_GQ] n_posterior;
    vector[N_contexts_GQ] g_posterior;
    vector[N_contexts_GQ] log10densb_posterior;
    vector[N_contexts_GQ] a_posterior;

    // generate data replication
    vector[N_GQ] log10_surf_rho_posterior;

    // factor level quantities
    for(i in 1:N_factors_GQ){
        log10densb_mean_posterior[i] = normal_rng(log10densb_mean_hyp1, log10densb_mean_hyp2);
        log10densb_std_posterior[i] = lower_trunc_normal_rng(0, log10densb_std_hyp, 0);
        g_lam_posterior[i] = rayleigh_rng(g_hyp);
        rb_sig_posterior[i] = rayleigh_rng(rb_hyp);
        n_mean_posterior[i] = trunc_normal_rng(n_mean_hyp1, n_mean_hyp2, 0, 15);
        n_std_posterior[i] = trunc_normal_rng(0, n_std_hyp2, 0, 15);
        a_sig_posterior[i] = rayleigh_rng(a_hyp);
        Re_sig_posterior[i] = rayleigh_rng(Re_hyp);
    }

    // context level quantities
    {
        vector[N_contexts_GQ] pre_term;
        vector[N_contexts_GQ] b_param;
        for(i in 1:N_contexts_GQ){
            log10densb_posterior[i] = trunc_normal_rng(log10densb_mean_posterior[factor_idx_GQ[i]], log10densb_std_posterior[factor_idx_GQ[i]], -5, 15);
            g_posterior[i] = trunc_exponential_rng(g_lam_posterior[factor_idx_GQ[i]], 0, 2);
            rb_posterior[i] = trunc_rayleigh_rng(rb_sig_posterior[factor_idx_GQ[i]], 0, 5);
            n_posterior[i] = trunc_normal_rng(n_mean_posterior[factor_idx_GQ[i]], n_std_posterior[factor_idx_GQ[i]], 0, 20);
            a_posterior[i] = rayleigh_rng(a_sig_posterior[factor_idx_GQ[i]]);
            Re_posterior[i] = trunc_rayleigh_rng(Re_sig_posterior[factor_idx_GQ[i]], 0, 20);
        }
        // some helper quantities
        b_param = sersic_b_parameter(n_posterior);
        pre_term = graham_preterm(g_posterior, a_posterior, n_posterior, b_param, rb_posterior, Re_posterior);

        // push forward data
        vector[N_GQ] mean_gsd = graham_surf_density_vec(
                                        R_GQ,
                                        pre_term[context_idx_GQ],
                                        g_posterior[context_idx_GQ],
                                        a_posterior[context_idx_GQ],
                                        rb_posterior[context_idx_GQ],
                                        n_posterior[context_idx_GQ],
                                        b_param[context_idx_GQ],
                                        Re_posterior[context_idx_GQ],
                                        log10densb_posterior[context_idx_GQ]);
        for(i in 1:N_GQ){
            log10_surf_rho_posterior[i] = trunc_normal_rng(mean_gsd[i], err_posterior[factor_idx_GQ[context_idx_GQ[i]]], -5, 15);
        }
    }

    // determine log likelihood function
    /*vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_calc[i], err[factor_idx[context_idx[i]]]);
    }*/
}

