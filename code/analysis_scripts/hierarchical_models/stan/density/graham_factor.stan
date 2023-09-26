functions {
    #include funcs_graham.stan
}


data {
    // Observations
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    array[N_tot] real<lower=0.0> R;
    // array of surface density values
    array[N_tot] real<lower=0.0> log10_surf_rho;

    // Individual Contexts
    // total number of individual contexts
    int<lower=1> N_contexts;
    // indexing of observations to context level
    array[N_tot] int<lower=1> context_idx;

    // Factor levels
    // number of factors
    int<lower=1> N_factors;
    // indexing of contexts to factor level
    array[N_contexts] int<lower=1> factor_idx;

    // progenitor break radius
    real<lower=0> rb_0;
    // kick velocity magnitudes normalised to escape velocity
    array[N_factors] real<lower=0> vkick_normed;

    // total number of out-of-sample points
    int<lower=1> N_OOS;
    // TODO do we need N_contexts_OOS and N_factors_OOS?
    // context ids for generated quantities
    array[N_OOS] int<lower=1> context_idx_OOS;
    // factor ids for generated quantities
    array[N_contexts] int<lower=1> factor_idx_OOS;
    // out-of-sample radii values
    array[N_OOS] real<lower=0, upper=max(R)> R_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
}


parameters {
    // latent parameters for each context
    array[N_contexts] real<lower=0> rb;
    array[N_contexts] real<lower=0> Re;
    array[N_contexts] real<lower=0, upper=20> n;
    array[N_contexts] real<lower=0> g;
    array[N_contexts] real log10densb;
    array[N_contexts] real<lower=0> a;

    // latent parameters for each factor
    array[N_factors] real<lower=0> Re_mean;
    array[N_factors] real<lower=0> Re_std;
    array[N_factors] real<lower=0> n_mean;
    array[N_factors] real<lower=0> n_std;
    array[N_factors] real<lower=0> g_mean;
    array[N_factors] real<lower=0> g_std;
    array[N_factors] real log10densb_mean;
    array[N_factors] real log10densb_std;
    array[N_factors] real<lower=0> a_mean;
    array[N_factors] real<lower=0> a_std;
    array[N_factors] real<lower=0> err;

    // global hyperpriors
    // for positive-constrained quantities, only need to worry about the 
    // variance. Suffix GM: global mean, GS: global std
    real<lower=0> Re_mean_GS;
    real<lower=0> Re_std_GS;
    real<lower=0> n_mean_GS;
    real<lower=0> n_std_GS;
    real<lower=0> g_mean_GS;
    real<lower=0> g_std_GS;
    real log10densb_mean_GM;
    real<lower=0> log10densb_mean_GS;
    real<lower=0> log10densb_std_GS;
    real<lower=0> a_mean_GS;
    real<lower=0> a_std_GS;

    // rb is a deterministic quantity
    // these parameters are for the fit
    real p;
    real q;
    real<lower=0> rb_err;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[14+N_factors] real lprior;
    lprior[1] = normal_lpdf(Re_mean_GS | 0, 20);
    lprior[2] = normal_lpdf(Re_std_GS | 0, 12);
    lprior[3] = normal_lpdf(n_mean_GS | 0, 16);
    lprior[4] = normal_lpdf(n_std_GS | 0, 10);
    lprior[5] = normal_lpdf(g_mean_GS | 0, 1);
    lprior[6] = normal_lpdf(g_std_GS | 0, 2);
    lprior[7] = normal_lpdf(log10densb_mean_GM | 8, 4);
    lprior[8] = normal_lpdf(log10densb_mean_GS | 0, 8);
    lprior[9] = normal_lpdf(log10densb_std_GS | 0, 2);
    lprior[10] = normal_lpdf(a_mean_GS | 0, 20);
    lprior[11] = normal_lpdf(a_std_GS | 0, 15);
    lprior[12] = normal_lpdf(rb_err | 0, 1);
    lprior[13] = normal_lpdf(p | 0, 1);
    lprior[14] = normal_lpdf(q | 0, 5);
    for(i in 1:N_factors){
        lprior[14+i] = normal_lpdf(err | 0, 1);
    }

    // deterministic rb
    array[N_contexts] real<lower=0> rb_calc;
    for(i in 1:N_contexts){
        rb_calc[i] = core_radius(vkick_normed[factor_idx[i]], rb_0, p, q);
    }
}


model {
    // density at hyperparameters
    target += sum(lprior);

    // connect hyperparameters to factor parameters
    target += normal_lpdf(Re_mean | 0, Re_mean_GS);
    target += normal_lpdf(Re_std | 0, Re_std_GS);
    target += normal_lpdf(n_mean | 0, n_mean_GS);
    target += normal_lpdf(n_std | 0, n_std_GS);
    target += normal_lpdf(g_mean | 0, g_mean_GS);
    target += normal_lpdf(g_std | 0, g_std_GS);
    target += normal_lpdf(log10densb_mean | log10densb_mean_GM, log10densb_mean_GS);
    target += normal_lpdf(log10densb_std | 0, log10densb_std_GS);
    target += normal_lpdf(a_mean | 0, a_mean_GS);
    target += normal_lpdf(a_std | 0, a_std_GS);

    target += normal_lpdf(log10(rb) | log10(rb_calc), rb_err);

    // connect factor parameters to context parameters
    target += normal_lpdf(Re | Re_mean, Re_std);
    target += normal_lpdf(n | n_mean, n_std);
    target += normal_lpdf(g | g_mean, g_std);
    target += normal_lpdf(log10densb | log10densb_mean, log10densb_std);
    target += normal_lpdf(a | a_mean, a_std);

    array[N_contexts] real pre_term;
    array[N_contexts] real b_param;
    for(i in 1:N_contexts){
        b_param[i] = sersic_b_parameter(n[i]);
        pre_term[i] = graham_preterm(g[i], a[i], n[i], b_param[i], rb[i], Re[i]);
    }

    // surface density calculation
    array[N_tot] real log10_surf_rho_calc;
    for(i in 1:N_tot){
        log10_surf_rho_calc[i] = graham_surf_density(
                                    R[i],
                                    pre_term[context_idx[i]],
                                    g[context_idx[i]],
                                    a[context_idx[i]],
                                    rb[context_idx[i]],
                                    b_param[context_idx[i]],
                                    Re[context_idx[i]],
                                    log10densb[context_idx[i]]
        );
    }
    target += normal_lpdf(log10_surf_rho | log10_surf_rho_calc, err[factor_idx[context_idx]]);
}


generated quantities {
    // posterior parameters for each factor level
    array[N_factors] real Re_mean_posterior;
    array[N_factors] real Re_std_posterior;
    array[N_factors] real n_mean_posterior;
    array[N_factors] real n_std_posterior;
    array[N_factors] real g_mean_posterior;
    array[N_factors] real g_std_posterior;
    array[N_factors] real log10densb_mean_posterior;
    array[N_factors] real log10densb_std_posterior;
    array[N_factors] real a_mean_posterior;
    array[N_factors] real a_std_posterior;

    // posterior parameters for each context
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
        Re_mean_posterior[i] = lower_trunc_norm_rng(0, Re_mean_GS, 0.);
        Re_std_posterior[i] = lower_trunc_norm_rng(0, Re_std_GS, 0.);
        n_mean_posterior[i] = trunc_norm_rng(0, n_mean_GS);
        n_std_posterior[i] = lower_trunc_norm_rng(0, n_std_GS, 0.);
        g_mean_posterior[i] = lower_trunc_norm_rng(0, g_mean_GS, 0.);
        g_std_posterior[i] = lower_trunc_norm_rng(0, g_std_GS, 0.);
        log10densb_mean_posterior[i] = normal_rng(log10densb_mean_GM, log10densb_mean_GS);
        log10densb_std_posterior[i] = lower_trunc_norm_rng(0, log10densb_std_GS);
        a_mean_posterior[i] = lower_trunc_norm_rng(0, a_mean_GS);
        a_std_posterior[i] = lower_trunc_norm_rng(0, a_std_GS);
    }

    // context level quantities
    {
        array[N_contexts] real pre_term;
        array[N_contexts] real b_param;
        for(i in 1:N_contexts){
            log10(rb_posterior) = normal_rng(log10(rb_calc[i]), rb_err[factor_idx[i]]);
            Re_posterior[i] = lower_trunc_norm_rng(Re_mean[factor_idx[i]], Re_std[factor_idx[i]], 0.);
            n_posterior[i] = trunc_norm_rng(n_mean[factor_idx[i]], n_std[factor_idx[i]], 0., 20.);
            g_posterior[i] = lower_trunc_norm_rng(g_mean[factor_idx[i]], g_std[factor_idx[i]], 0.);
            log10densb_posterior[i] = lower_trunc_norm_rng(log10densb_mean[factor_idx[i]], log10densb_std[factor_idx[i]], 0.);
            a_posterior[i] = lower_trunc_norm_rng(a_mean[factor_idx[i]], a_std[factor_idx[i]], 0.);

            b_param[i] = sersic_b_parameter(n_posterior[i]);
            pre_term[i] = graham_preterm(g_posterior[i], a_posterior[i], n_posterior[i], b_param[i], rb[i], Re_posterior[i]);
        }

        for(i in 1:N_GQ){
            log10_surf_rho_posterior[i] = normal_rng(graham_surf_density(
                                            R_OOS[i],
                                            pre_term[context_idx_GQ[i]],
                                            g_posterior[context_idx_GQ[i]],
                                            a_posterior[context_idx_GQ[i]],
                                            rb_posterior[context_idx_GQ[i]],
                                            b_param[context_idx_GQ[i]],
                                            Re_posterior[context_idx_GQ[i]],
                                            log10densb_posterior[context_idx_GQ[i]]
                                            ), err[factor_idx[context_idx_GQ[i]]]);
        }
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i[i], err[factor_idx[context_idx[i]]]);
    }
}

