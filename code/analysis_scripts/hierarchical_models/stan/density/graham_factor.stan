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

    // progenitor break radius
    real<lower=0> rb_0;
    // kick velocity magnitudes normalised to escape velocity
    array[N_factors] real<lower=0> vkick_normed;

    // Out of Sample points
    // follows same structure as above
    // total number of OOS points
    int<lower=1> N_OOS;
    // OOS radii values
    array[N_OOS] real<lower=0, upper=max(R)> R_OOS;

    // OOS contexts
    // total number of individual OOS contexts
    //int<lower=1> N_contexts_OOS;
    // context ids for generated quantities
    array[N_OOS] int<lower=1> context_idx_OOS;

    // OOS factors
    // total number of individual OOS factors
    //int<lower=1> N_factors_OOS;
    // factor ids for generated quantities
    array[N_contexts] int<lower=1> factor_idx_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
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

    // parameters for each factor level
    array[N_factors] real log10densb_mean;
    array[N_factors] real<lower=0> log10densb_std;
    array[N_factors] real<lower=0> g_lam;
    array[N_factors] real<lower=0> rb_sig;
    array[N_factors] real<lower=0, upper=15> n_mean;
    array[N_factors] real<lower=0, upper=15> n_std;
    array[N_factors] real<lower=0> a_sig;
    array[N_factors] real<lower=0> Re_sig;
    array[N_factors] real<lower=0> err;

    // latent parameters for each context
    array[N_contexts] real<lower=0, upper=5> rb;
    array[N_contexts] real<lower=0, upper=20> Re;
    array[N_contexts] real<lower=0, upper=20> n;
    array[N_contexts] real<lower=0, upper=2> g;
    array[N_contexts] real<lower=-5, upper=15> log10densb;
    array[N_contexts] real<lower=0> a;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[11+N_factors] real lprior;
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
    for(i in 1:N_factors){
        lprior[11+i] = normal_lpdf(err | 0, 0.1);
    }

    // pre definition of calculated density for log-likelihood calculation
    // in generated quantities block
    array[N_tot] real<lower=-5, upper=15> log10_surf_rho_calc;
    {
        // no need to track these helper variables, so put in private scope
        array[N_contexts] real pre_term;
        array[N_contexts] real b_param;
        for(i in 1:N_contexts){
            b_param[i] = sersic_b_parameter(n[i]);
            pre_term[i] = graham_preterm(g[i], a[i], n[i], b_param[i], rb[i], Re[i]);
        }
        // surface density calculation
        for(i in 1:N_tot){
            log10_surf_rho_calc[i] = graham_surf_density(
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
        }
    }
}


model {
    // density at hyperparameters
    target += sum(lprior);

    // connect hyperparameters to factor parameters
    for(i in 1:N_factors){
        target += normal_lpdf(log10densb_mean[i] | log10densb_mean_hyp1, log10densb_mean_hyp2);
        target += normal_lpdf(log10densb_std[i] | 0, log10densb_std_hyp);
        target += rayleigh_lpdf(g_lam[i] | g_hyp);
        target += rayleigh_lpdf(rb_sig[i] | rb_hyp);
        target += normal_lpdf(n_mean[i] | n_mean_hyp1, n_mean_hyp2);
        target += normal_lpdf(n_std[i] | n_std_hyp1, n_std_hyp2);
        target += rayleigh_lpdf(a_sig[i] | a_hyp);
        target += rayleigh_lpdf(Re_sig[i] | Re_hyp);
    }

    // connect factor parameters to context parameters
    for(i in 1:N_contexts){
        target += normal_lpdf(log10densb[i] | log10densb_mean, log10densb_std);
        target += exponential_lpdf(g[i] | g_lam[factor_idx[i]]);
        target += rayleigh_lpdf(rb[i] | rb_sig[factor_idx[i]]);
        target += normal_lpdf(n[i] | n_mean[factor_idx[i]], n_std[factor_idx[i]]);
        target += rayleigh_lpdf(a[i] | a_sig[factor_idx[i]]);
        target += rayleigh_lpdf(Re[i] | Re_sig[factor_idx[i]]);
    }

    target += normal_lpdf(log10_surf_rho | log10_surf_rho_calc, err[factor_idx[context_idx]]);
}


generated quantities {
    // parameters for each factor level
    array[N_factors] real log10densb_mean_posterior;
    array[N_factors] real log10densb_std_posterior;
    array[N_factors] real g_lam_posterior;
    array[N_factors] real rb_sig_posterior;
    array[N_factors] real n_mean_posterior;
    array[N_factors] real n_std_posterior;
    array[N_factors] real a_sig_posterior;
    array[N_factors] real Re_sig_posterior;

    // parameters for each context
    array[N_contexts] real rb_posterior;
    array[N_contexts] real Re_posterior;
    array[N_contexts] real n_posterior;
    array[N_contexts] real g_posterior;
    array[N_contexts] real log10densb_posterior;
    array[N_contexts] real a_posterior;

    // generate data replication
    array[N_GQ] real R_GQ = append_array(R, R_OOS);
    array[N_GQ] int context_idx_GQ = append_array(context_idx, context_idx_OOS);
    array[N_GQ] real log10_surf_rho_posterior;

    // factor level quantities
    for(i in 1:N_factors){
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
        array[N_contexts] real pre_term;
        array[N_contexts] real b_param;
        for(i in 1:N_contexts){
            log10densb_posterior[i] = trunc_normal_rng(log10densb_mean_posterior[factor_idx[i]], log10densb_std_posterior[factor_idx[i]], -5, 15);
            g_posterior[i] = trunc_exponential_rng(g_lam_posterior[factor_idx[i]], 0, 2);
            rb_posterior[i] = trunc_rayleigh_rng(rb_sig_posterior[factor_idx[i]], 0, 5);
            n_posterior[i] = trunc_normal_rng(n_mean_posterior[factor_idx[i]], n_std_posterior[factor_idx[i]], 0, 20);
            a_posterior[i] = rayleigh_rng(a_sig_posterior[factor_idx[i]]);
            Re_posterior[i] = trunc_rayleigh_rng(Re_sig_posterior[factor_idx[i]], 0, 20);
            // some helper quantities
            b_param[i] = sersic_b_parameter(n[i]);
            pre_term[i] = graham_preterm(g[i], a[i], n[i], b_param[i], rb[i], Re[i]);
        }

        // push forward data
        real mean_gsd;
        for(i in 1:N_GQ){
            mean_gsd = graham_surf_density(
                            R_OOS[i],
                            pre_term[context_idx_GQ[i]],
                            g_posterior[context_idx_GQ[i]],
                            a_posterior[context_idx_GQ[i]],
                            rb_posterior[context_idx_GQ[i]],
                            n_posterior[context_idx_GQ[i]],
                            b_param[context_idx_GQ[i]],
                            Re_posterior[context_idx_GQ[i]],
                            log10densb_posterior[context_idx_GQ[i]]
                        );
            log10_surf_rho_posterior[i] = trunc_normal_rng(mean_gsd, err[factor_idx[context_idx_GQ[i]]], -5, 15);
        }
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_calc[i], err[factor_idx[context_idx[i]]]);
    }
}

